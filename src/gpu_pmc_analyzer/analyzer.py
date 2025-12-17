import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from itertools import combinations

import numpy as np
import pandas as pd

from src.utils import logger


@dataclass
class AnalyzerConfig:
    """Settings for kernel pattern detection."""

    matmul_role_names: List[str] = field(
        default_factory=lambda: ["QKV", "Wo", "FFN1", "FFN2"]
    )
    matmul_target_flops_ratio: List[float] = field(
        default_factory=lambda: [3.0, 1.0, 4.0, 4.0]
    )

    # Attention must be located between these two roles (exclusive)
    attention_left_role: str = "QKV"
    attention_right_role: str = "Wo"

    # elementwise_add must be located between these two roles (exclusive)
    elementwise_add_left_role: str = "Wo"
    elementwise_add_right_role: str = "FFN1"
    family: str = "gpt2"

    @classmethod
    def for_family(cls, family: Optional[str]) -> "AnalyzerConfig":
        """
        Factory for family-specific kernel role settings.

        Keep this as a simple registry so adding a new family is a one-liner.
        """
        fam = (family or "gpt2").lower()
        family_kwargs: Dict[str, Dict[str, object]] = {
            "gpt2": {
                "matmul_role_names": ["QKV", "Wo", "FFN1", "FFN2"],
                "matmul_target_flops_ratio": [3.0, 1.0, 4.0, 4.0],
                "attention_left_role": "QKV",
                "attention_right_role": "Wo",
                "elementwise_add_left_role": "Wo",
                "elementwise_add_right_role": "FFN1",
                "family": "gpt2",
            },
            # LLaMA/Qwen share the same role layout; only "family" differs.
            "llama": {
                "matmul_role_names": ["Q", "K", "V", "Wo", "FFN1", "FFN2", "FFN3"],
                "matmul_target_flops_ratio": [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
                "attention_left_role": "V",
                "attention_right_role": "Wo",
                "elementwise_add_left_role": "Wo",
                "elementwise_add_right_role": "FFN1",
                "family": "llama",
            },
            "qwen": {
                "matmul_role_names": ["Q", "K", "V", "Wo", "FFN1", "FFN2", "FFN3"],
                "matmul_target_flops_ratio": [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
                "attention_left_role": "V",
                "attention_right_role": "Wo",
                "elementwise_add_left_role": "Wo",
                "elementwise_add_right_role": "FFN1",
                "family": "qwen",
            },
        }

        kwargs = family_kwargs.get(fam)
        if kwargs is None:
            logger.warning(f"Unknown family '{family}', defaulting to gpt2")
            kwargs = family_kwargs["gpt2"]
        return cls(**kwargs)


def normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v / norm


def build_feature_vectors(df: pd.DataFrame) -> np.ndarray:
    """
    Build 3D feature vectors:
      v(k) = [F(k), r_sh(k), B_tot(k)]
    """
    F = df["total_flops_with_tensor"].values
    R = df["shared_ratio"].values
    B = (
        df["global_op_ld_lookup_miss_bytes"] + df["global_op_st_lookup_miss_bytes"]
    ).values
    return np.vstack([F, R, B]).T


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def hybrid_similarity(
    dfA: pd.DataFrame, dfB: pd.DataFrame, f_weight: float = 0.90
) -> float:
    """
    Hybrid similarity: flops_weight * FLOPs similarity + (1 - flops_weight) * cosine similarity(F, r, B).
    """
    F1 = dfA["total_flops_with_tensor"].values.astype(float)
    F2 = dfB["total_flops_with_tensor"].values.astype(float)

    flops_err = np.linalg.norm(F1 - F2)
    flops_norm = np.linalg.norm(F1) + 1e-8
    flops_sim = 1.0 - flops_err / flops_norm

    V1 = normalize_vec(build_feature_vectors(dfA))
    V2 = normalize_vec(build_feature_vectors(dfB))
    cos_sim = np.sum(V1 * V2)

    return f_weight * flops_sim + (1 - f_weight) * cos_sim


class Analyzer:
    @classmethod
    def from_df(cls, df: pd.DataFrame, config: AnalyzerConfig) -> "Analyzer":
        """
        Create an Analyzer from an in-memory dataframe (used by two-phase).
        """
        obj = cls.__new__(cls)
        obj.cfg = config
        obj.data = df
        obj.layer_T = None
        obj.layer_idx_pool = []
        obj.last_token_cycles = []
        return obj

    def __init__(
        self,
        df_path: str,
        config: AnalyzerConfig,
    ):
        self.cfg = config
        self.data = self._load_feature_csv(df_path)
        self.layer_T: Optional[int] = None
        # Pool of all start/end indices of layers (absolute indices in self.data)
        self.layer_idx_pool: List[Tuple[int, int]] = []
        # Last detected token cycle (start,end) indices in self.data
        self.last_token_cycles: List[Tuple[int, int]] = []

    def _load_feature_csv(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading processed feature CSV: {path}")
        df = pd.read_csv(path)

        required = [
            "Kernel Name",
            "total_flops_with_tensor",
            "shared_ratio",
            "global_op_ld_lookup_miss_bytes",
            "global_op_st_lookup_miss_bytes",
            "accumulated_time",
        ]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        return df

    def detect_layer_period_T(self, T_max: int = 64) -> int:
        def _find_best_period_T(V: np.ndarray, F: np.ndarray) -> Tuple[int, float]:
            def _longest_true_run(mask: np.ndarray) -> int:
                """
                Return the longest consecutive True run length in a 1D boolean array.
                """
                if mask.size == 0:
                    return 0
                # Convert to int and compute run lengths
                m = mask.astype(np.int8)
                best = 0
                cur = 0
                for v in m:
                    if v:
                        cur += 1
                        if cur > best:
                            best = cur
                    else:
                        cur = 0
                return int(best)

            best_T = None
            best_score = -1e18
            scores: Dict[int, float] = {}

            upper = min(T_max, max(2, len(V) // 2))

            for T in range(2, upper + 1):
                sim = cosine_similarity(V[:-T], V[T:])

                pair_flops = 0.5 * (F[:-T] + F[T:])
                w = np.sqrt(pair_flops + 1e-8)

                # Deduct score if F[:-T] and F[T:] are too different
                a = F[:-T]
                b = F[T:]
                d = a - b
                flops_err = np.linalg.norm(d)
                flops_norm = np.linalg.norm(a) + np.linalg.norm(b) + 1e-8
                flops_sim = 1.0 - flops_err / flops_norm

                # Reward long consecutive near-zero segments in (F[:-T] - F[T:])
                zero_mask = np.abs(d) <= 1
                longest_run = _longest_true_run(zero_mask)
                longest_norm = float(longest_run / max(1, zero_mask.size))

                if self.cfg.family != "gpt2":
                    score = float(np.sum(sim * w) / (np.sum(w) + 1e-8)) * (
                        flops_sim + 0.5 * longest_norm * 10
                    )
                else:
                    score = float(np.sum(sim * w) / (np.sum(w) + 1e-8)) * (flops_sim)
                scores[int(T)] = float(score)
                if score > best_score:
                    best_score = score
                    best_T = T

            # If a larger T wins only because it is a multiple of a smaller "fundamental" period,
            # prefer the smallest divisor whose score is close to the best score.
            if best_T is not None:
                fundamental_tol = 0.1  # allow up to 10% score drop to prefer smaller fundamental period
                divisors = [
                    t
                    for t in range(2, int(best_T))
                    if (int(best_T) % t == 0)
                    and (t in scores)
                    and (scores[t] >= best_score * (1.0 - fundamental_tol))
                ]
                if divisors:
                    best_T = min(divisors)
                    best_score = scores[int(best_T)]

            return best_T, best_score

        V = normalize_vec(build_feature_vectors(self.data))
        F = self.data["total_flops_with_tensor"].values.astype(float)
        T, score = _find_best_period_T(V, F)
        assert T is not None
        self.layer_T = int(T)
        logger.success(f"best T={T}, score={score:.3e}")
        return int(T)

    def find_start_point(
        self, df: pd.DataFrame, T: int, max_search_s: int = 256
    ) -> int:
        """
        Find start point s such that df[s:s+T] and df[s+T:s+2T] are most similar.
        This is the "start point" logic you referenced.
        """

        def _find_plateau_start(arr, rel_eps=1e-3):
            arr = np.asarray(arr)
            n = len(arr)
            max_len = 0
            start_idx = None
            i = 0
            while i < n:
                base = arr[i]
                j = i + 1

                while j < n and abs(arr[j] - base) / abs(base) < rel_eps:
                    j += 1

                if j - i >= max_len:
                    max_len = j - i
                    start_idx = i

                i = j
            return int(start_idx)

        N = len(df)
        sims = []
        flops_sums = []
        valid_s = []

        for s in range(max_search_s):
            if s + 2 * T > N:
                break
            L0 = df.iloc[s : s + T]
            L1 = df.iloc[s + T : s + 2 * T]
            sims.append(hybrid_similarity(L0, L1))
            flops_sums.append(float(L0["total_flops_with_tensor"].sum()))
            valid_s.append(s)

        sims = np.array(sims)
        flops_sums = np.array(flops_sums)
        valid_s = np.array(valid_s)

        sim_max = np.max(sims)
        sim_threshold = sim_max * 0.95
        flops_max = np.percentile(flops_sums, 90)
        flops_threshold = flops_max * 0.9

        valid_mask = (sims > sim_threshold) & (flops_sums > flops_threshold)

        if not valid_mask.any():
            chosen_s = _find_plateau_start(sims)
        else:
            chosen_s = valid_s[np.argmax(valid_mask)]

        flops_log = np.log(np.array(flops_sums) + 1)
        flops_thr = np.exp(flops_log.mean() / 1.9) * 2

        while float(df.iloc[chosen_s]["total_flops_with_tensor"]) < flops_thr:
            chosen_s += 1
        return chosen_s

    def compute_layer_count_from_start(
        self,
        df: pd.DataFrame,
        *,
        start: int,
        T: int,
        eps_factor: float = 0.93,
    ) -> int:
        """
        Count repeating layers starting at `start` under period T.
        """
        V = normalize_vec(build_feature_vectors(df))
        N = len(df)

        first = V[start : start + T]
        second = V[start + T : start + 2 * T]
        base_sim = float(np.mean(cosine_similarity(first, second)))
        thr = base_sim * eps_factor

        layer_count = 2
        i = 1
        while True:
            s_i = start + i * T
            s_next = start + (i + 1) * T
            if s_next + T > N:
                break

            Li = V[s_i : s_i + T]
            Li_next = V[s_next : s_next + T]
            sim_i = float(np.mean(cosine_similarity(Li, Li_next)))
            if sim_i >= thr:
                layer_count += 1
            else:
                break
            i += 1

        return layer_count

    def peel_token_cycles(
        self,
        df_peeled: pd.DataFrame,
        *,
        T: int,
        layer_count: int,
        max_search_s: int = 256,
        absolute_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        """
          - find start point on current df
          - end = start + layer_count*T
          - drop everything before end
          - repeat
        """
        token_len = layer_count * T
        cycles: List[Tuple[int, int]] = []
        offset = absolute_offset
        cur = df_peeled.reset_index(drop=True)

        while True:
            if len(cur) < 2 * T:
                break
            s = self.find_start_point(cur, T, max_search_s=max_search_s)
            e = s + token_len
            logger.info(f"Token cycle {len(cycles) + 1}: [{offset + s}:{offset + e})")
            if e > len(cur):
                break
            cycles.append((offset + s, offset + e))
            cur = cur.iloc[e:].reset_index(drop=True)
            offset += e

        return cycles

    def extract_first_cycle(self, max_search_s: int = 256) -> pd.DataFrame:
        """
        Extract the main cycle from the data.

        Args:
            max_search_s: Restrict the search to the first max_search_s steps to save time.
        """
        T = self.layer_T
        if T is None:
            raise ValueError("layer_T is None; call detect_layer_period_T() first")

        chosen_s = self.find_start_point(self.data, T, max_search_s=max_search_s)

        # Reset pool and store the first layer interval
        self.layer_idx_pool = [(chosen_s, chosen_s + T)]
        logger.success(f"[Cycle Detection] Final start = {chosen_s}, T = {T}")

        return self.data.iloc[chosen_s : chosen_s + T].reset_index(drop=True)

    def compute_layer_count(self, eps_factor: float = 0.93) -> int:
        T = self.layer_T
        if T is None:
            raise ValueError("layer_T is None; call detect_layer_period_T() first")
        if not self.layer_idx_pool:
            raise ValueError(
                "layer_idx_pool is empty; call extract_first_cycle() first"
            )

        start0 = int(self.layer_idx_pool[0][0])
        N = len(self.data)

        V = normalize_vec(build_feature_vectors(self.data))

        first = V[start0 : start0 + T]
        second = V[start0 + T : start0 + 2 * T]

        base_sim = np.mean(cosine_similarity(first, second))
        thr = base_sim * eps_factor

        logger.info("========== Layer Similarity Debug ==========")
        logger.info(f"start={start0}, T={T}, base_sim={base_sim:.3f}, thr={thr:.3f}")
        logger.info("---------------------------------------------")
        logger.info(f"Layer 0: [{start0}:{start0 + T})")

        # Start with 2 layers: Layer 0 and Layer 1 are already confirmed similar
        # Rebuild pool to include all validated layers
        self.layer_idx_pool = [(start0, start0 + T), (start0 + T, start0 + 2 * T)]
        layer_count = 2
        i = 1

        while True:
            s_i = start0 + i * T
            s_next = start0 + (i + 1) * T
            if s_next + T > N:
                break

            Li = V[s_i : s_i + T]
            Li_next = V[s_next : s_next + T]

            sim_i = np.mean(cosine_similarity(Li, Li_next))

            if sim_i >= thr:
                layer_count += 1  # Add Layer i+1
                self.layer_idx_pool.append((s_next, s_next + T))
            else:
                logger.warning(f"STOP at layer {i}: sim={sim_i:.3f} < thr={thr:.3f}")
                break

            i += 1

        logger.info("=============================================")
        return layer_count

    def detect_max_token_cycles(
        self,
        eps_factor: float = 0.93,
        process_token: Optional[int] = None,
        processed_path: str = "processed.csv",
    ) -> int:
        """
        Detect multiple small cycles (token positions, maxtoken).

        Each small cycle corresponds to one token position and contains
        all layers (layer_count layers, each layer is T kernels long).
        After finding one cycle, detect the next cycle starting from its end.
        """
        # First, compute layer count to know the length of one token cycle
        layer_count = self.compute_layer_count(eps_factor=eps_factor)

        T = self.layer_T
        assert T is not None
        N = len(self.data)

        V = normalize_vec(build_feature_vectors(self.data))

        # Each token cycle contains all layers: length = layer_count * T
        token_cycle_length = layer_count * T

        logger.info("========== Token Cycle Detection ==========")
        logger.info(f"T={T}, layer_count={layer_count}")
        logger.info(f"Token cycle length = {token_cycle_length} (layers * T)")
        logger.info("---------------------------------------------")

        token_cycles = []
        start0 = int(self.layer_idx_pool[0][0]) if self.layer_idx_pool else 0
        current_start = start0

        # Add the first token cycle (already detected)
        if current_start + token_cycle_length <= N:
            if self._verify_token_cycle(V, current_start, T, layer_count, eps_factor):
                token_cycles.append((current_start, current_start + token_cycle_length))
                logger.info(
                    f"Token cycle 1: [{current_start}:{current_start + token_cycle_length})"
                )
                current_start = current_start + token_cycle_length

        # Detect subsequent token cycles
        while current_start + token_cycle_length <= N:
            # Detect the start of next token cycle using surge detection
            cycle_start = self._find_next_cycle_start(
                self.data, current_start, T, max_search_s=256
            )

            if cycle_start is None or cycle_start + token_cycle_length > N:
                break

            # Verify this is a valid token cycle by checking layer similarity
            cycle_end = cycle_start + token_cycle_length
            if self._verify_token_cycle(V, cycle_start, T, layer_count, eps_factor):
                token_cycles.append((cycle_start, cycle_end))
                current_start = cycle_end  # Start next search from end of this cycle
            else:
                # If verification fails, try next position
                current_start = cycle_start + T

        logger.info("=============================================")
        logger.info(f"Total token cycles (maxtoken) = {len(token_cycles)}")
        # Save for debugging/inspection
        self.last_token_cycles = list(token_cycles)

        if not token_cycles:
            logger.warning("Warning: No token cycles detected; skip export")
            return 0

        # process the specified token cycle
        idx = 0 if process_token is None else process_token
        idx = max(0, min(idx, len(token_cycles) - 1))
        start, end = token_cycles[idx]

        slice_df = self.data.iloc[start:end].reset_index(drop=True)
        averaged_slice = self._average_layers_by_period(slice_df, T, layer_count)
        roles_df = self.assign_roles(averaged_slice)

        out_dir = os.path.dirname(processed_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        roles_df.to_csv(processed_path, index=False)
        logger.success(f"Saved token cycle #{idx} → {processed_path}")

        return len(token_cycles)

    def _average_layers_by_period(
        self, df: pd.DataFrame, T: int, layer_count: int
    ) -> pd.DataFrame:
        """
        Average kernels across repeated layers: position i is mean of i + k*T.

        Kernel-name (and other non-numeric fields) are kept from the first layer.
        """
        if layer_count <= 1:
            return df.reset_index(drop=True)

        expected_len = layer_count * T
        if len(df) < expected_len:
            logger.warning(
                f"Data shorter than expected for averaging: len={len(df)}, expected={expected_len}"
            )

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "index"]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        averaged_rows = []
        for i in range(T):
            idxs = [i + k * T for k in range(layer_count) if i + k * T < len(df)]
            if not idxs:
                continue

            group = df.iloc[idxs]
            averaged = {col: group.iloc[0][col] for col in non_numeric_cols}
            for col in numeric_cols:
                averaged[col] = group[col].mean()

            averaged_rows.append(averaged)

        return pd.DataFrame(averaged_rows)

    def _average_token_cycle(
        self, df: pd.DataFrame, T: int, layer_count: int
    ) -> pd.DataFrame:
        """
        Average kernels across token cycle: position i is mean of i + k*T*layer_count.

        Kernel-name (and other non-numeric fields) are kept from the first layer.
        """
        working = self._add_debug_columns(df.copy())
        token_cycle_len = layer_count * T

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "index"]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        averaged_rows = []
        for i in range(token_cycle_len):
            idxs = []
            j = i
            while j < len(df):
                idxs.append(j)
                j += token_cycle_len

            if not idxs:
                continue

            group = df.iloc[idxs]
            averaged = {col: group.iloc[0][col] for col in non_numeric_cols}
            for col in numeric_cols:
                # Coefficient of variation to detect outliers in FLOPs
                if col == "total_flops_with_tensor" and working["is_matmul_candidate"].iloc[idxs[0]]:
                    mean = float(group[col].mean())
                    std = float(group[col].std())
                    cv = std / (abs(mean) + 1e-12)
                    if cv > 0.2:
                        logger.warning(
                            f"High CV detected in FLOPs during token-cycle averaging: CV={cv:.3g}, "
                            f"mean={mean:.3g}, std={std:.3g}, n={len(group)}"
                        )
                        import pdb; pdb.set_trace()
                averaged[col] = group[col].mean()

            averaged_rows.append(averaged)

        return pd.DataFrame(averaged_rows)

    def _find_next_cycle_start(
        self, df: pd.DataFrame, start_pos: int, T: int, max_search_s: int = 256
    ) -> Optional[int]:
        """
        Find the start of next token cycle using surge detection.
        Similar to extract_first_cycle but starting from a specific position.
        """
        N = len(df)

        sims = []
        flops_sums = []
        valid_s = []

        for s_offset in range(max_search_s):
            s = start_pos + s_offset
            if s + 2 * T > N:
                break

            L0 = df.iloc[s : s + T]
            L1 = df.iloc[s + T : s + 2 * T]

            sim_s = hybrid_similarity(L0, L1)
            flops_s = float(L0["total_flops_with_tensor"].sum())

            sims.append(sim_s)
            flops_sums.append(flops_s)
            valid_s.append(s)

        if len(sims) < 2:
            return valid_s[0] if valid_s else None

        sims = np.array(sims)
        flops_sums = np.array(flops_sums)
        valid_s = np.array(valid_s)

        # Detect surge with FLOPs constraint
        sim_max = np.max(sims)
        sim_threshold = sim_max * 0.95
        flops_max = np.max(flops_sums)
        flops_threshold = flops_max * 0.9

        naive_start = int(
            valid_s[np.argmax((sims > sim_threshold) & (flops_sums > flops_threshold))]
        )
        while int(df.iloc[naive_start]["total_flops_with_tensor"]) == 0:
            naive_start += 1
            logger.warning(
                f"Warning: Naive start {naive_start} has 0 FLOPs, incrementing"
            )
        return naive_start

    def _verify_token_cycle(
        self, V: np.ndarray, start: int, T: int, layer_count: int, eps_factor: float
    ) -> bool:
        """
        Verify that a segment is a valid token cycle by checking layer similarities.
        """
        N = len(V)
        token_cycle_length = layer_count * T

        if start + token_cycle_length > N:
            return False

        # Check that consecutive layers are similar
        for i in range(layer_count - 1):
            layer_start = start + i * T
            layer_next = start + (i + 1) * T

            if layer_next + T > N:
                return False

            Li = V[layer_start : layer_start + T]
            Li_next = V[layer_next : layer_next + T]

            sim = np.mean(cosine_similarity(Li, Li_next))

            # Use first layer pair as baseline
            if i == 0:
                base_sim = sim
                thr = base_sim * eps_factor

            if sim < thr:
                return False

        return True

    def _add_debug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add debug/helper columns used for matmul role detection.
        """
        working = df.copy()

        flops = working["total_flops_with_tensor"].values
        flops_log = np.log(flops + 1)
        mu, sigma = flops_log.mean(), flops_log.std()
        flops_thr = mu + 0.5 * sigma

        shared = working["shared_ratio"].values
        att_thr = 0.7

        working["flops_log"] = flops_log
        working["flops_threshold"] = flops_thr
        working["is_matmul_candidate"] = flops_log > flops_thr
        working["is_attention_candidate"] = shared > att_thr
        working["elementwise_add_fma_ops"] = (
            0  # Later add the FMA ops of elementwise_add
        )

        return working

    def assign_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign roles (matmul roles + Attention + elementwise_add) based on FLOPs prominence and target ratios.
        """
        working = self._add_debug_columns(df).reset_index(drop=True)

        role_names = list(self.cfg.matmul_role_names)
        target = np.array(self.cfg.matmul_target_flops_ratio, dtype=float)

        cand = working[working["is_matmul_candidate"]]
        matmul_indices = list(cand.index.values)
        need = len(role_names)
        if len(matmul_indices) < need:
            logger.warning(
                "Warning: Not enough matmul candidates, assigning unknown roles"
            )
            working["role"] = "unknown"
            return working

        logger.info(
            f"Assigning roles: need={need} from {len(matmul_indices)} matmul candidates "
            f"(attention between {self.cfg.attention_left_role}->{self.cfg.attention_right_role}, "
            f"elementwise_add between {self.cfg.elementwise_add_left_role}->{self.cfg.elementwise_add_right_role})"
        )
        cand_flops = working["total_flops_with_tensor"]
        total_cycle_flops = float(cand_flops.sum()) + 1e-8

        target_norm = target / (target.sum() + 1e-8)

        best_err = 1e18
        best_group = None
        best_att_idx = None
        best_add_idx = None
        best_add2_idx = None

        for combo in combinations(sorted(matmul_indices), need):
            combo = tuple(combo)
            combo_set = set(combo)
            role_to_idx: Dict[str, int] = dict(zip(role_names, combo))

            left_idx = role_to_idx[self.cfg.attention_left_role]
            right_idx = role_to_idx[self.cfg.attention_right_role]
            if left_idx >= right_idx:
                continue

            # Attention: between (left_idx, right_idx)
            att_region = working.iloc[left_idx + 1 : right_idx]
            att_candidates = att_region[att_region["is_attention_candidate"]]
            if att_candidates.empty:
                continue
            att_idx = att_candidates["total_flops_with_tensor"].idxmax()
            if att_idx in combo_set:
                continue

            # elementwise_add: between (add_left, add_right)
            add_left = role_to_idx[self.cfg.elementwise_add_left_role]
            add_right = role_to_idx[self.cfg.elementwise_add_right_role]
            if add_left >= add_right:
                continue
            add_region = working.iloc[add_left + 1 : add_right]
            add_candidates = add_region[
                (add_region["total_fma_ops"] > 0)
                & (
                    np.isclose(
                        add_region["total_flops_with_tensor"],
                        2 * add_region["total_fma_ops"],
                    )
                )
            ]
            if add_candidates.empty:
                continue
            add_idx = int(add_candidates.index.min())
            if add_idx in combo_set or add_idx == att_idx:
                continue

            # elementwise_add after the last FFN matmul (same rule as above)
            last_ffn_idx = role_to_idx[role_names[-1]]
            add2_region = working.iloc[last_ffn_idx + 1 :]
            add2_candidates = add2_region[
                (add2_region["total_fma_ops"] > 0)
                & (
                    np.isclose(
                        add2_region["total_flops_with_tensor"],
                        2 * add2_region["total_fma_ops"],
                    )
                )
            ]
            add2_idx = None
            if not add2_candidates.empty:
                add2_idx = int(add2_candidates.index.min())
                if add2_idx in combo_set or add2_idx == att_idx or add2_idx == add_idx:
                    add2_idx = None

            combo_flops = cand_flops.loc[list(combo)].values.astype(float)
            flops_norm = combo_flops / (combo_flops.sum() + 1e-8)

            # Base error: match target FLOPs ratios among matmul roles
            err = float(np.sum((flops_norm - target_norm) ** 2))

            # Coverage penalty: ensure QKV/Wo/FFN matmuls dominate total cycle FLOPs
            coverage = float(combo_flops.sum()) / total_cycle_flops
            err += (1 - coverage) ** 2
            if err < best_err:
                best_err = err
                best_group = combo
                best_att_idx = att_idx
                best_add_idx = add_idx
                best_add2_idx = add2_idx

        if best_group is None or best_att_idx is None or best_add_idx is None:
            logger.warning(
                "Warning: Unable to find role combination, assigning unknown roles"
            )
            working["role"] = "unknown"
            return working

        role_to_idx = dict(zip(role_names, best_group))
        att_idx = int(best_att_idx)
        add_idx = int(best_add_idx)
        add2_idx = int(best_add2_idx) if best_add2_idx is not None else None

        working["role"] = "other"
        for r in role_names:
            working.at[role_to_idx[r], "role"] = r
        working.at[att_idx, "role"] = "Attention"
        working.at[add_idx, "role"] = "elementwise_add"
        if add2_idx is not None:
            working.at[add2_idx, "role"] = "elementwise_add"

        working.at[add_idx, "elementwise_add_fma_ops"] = working.at[
            add_idx, "total_fma_ops"
        ]
        if add2_idx is not None:
            working.at[add2_idx, "elementwise_add_fma_ops"] = working.at[
                add2_idx, "total_fma_ops"
            ]

        # Assign splitKreduce role after role assignment
        self._assign_splitkreduce_role(working, add_idx, role_to_idx, role_names)

        return working

    def _assign_splitkreduce_role(
        self,
        working: pd.DataFrame,
        elementwise_add_idx: int,
        role_to_idx: Dict[str, int],
        role_names: List[str],
    ) -> None:
        """
        Assign splitKreduce role to kernels that follow matmul kernels with split-K pattern.
        """
        # Get reference value from elementwise_add global write
        reference_value = working.at[
            elementwise_add_idx, "global_op_st_lookup_miss_bytes"
        ]
        logger.info(
            f"splitKreduce assignment: Using elementwise_add global write as reference: {reference_value}"
        )

        if reference_value == 0:
            logger.warning("splitKreduce assignment: Reference value is 0, skipping")
            return

        # Matmul roles that can have split-K pattern (Q, K, V, FFN1, FFN2, FFN3, etc.)
        matmul_roles = self.cfg.matmul_role_names

        splitkreduce_assigned = []

        # Check each matmul kernel
        for role in matmul_roles:
            if role not in role_to_idx:
                continue

            matmul_idx = role_to_idx[role]
            matmul_global_write = working.at[
                matmul_idx, "global_op_st_lookup_miss_bytes"
            ]

            # Calculate reference value based on current matmul role
            # If role is QKV, use 3 * reference_value
            # If role is FFN1 or FFN2 (when FFN3 exists), use 2 * reference_value
            # Otherwise use 1 * reference_value
            if role == "QKV":
                current_reference_value = 3 * reference_value
            elif role == "FFN1":
                current_reference_value = 2 * reference_value
            elif role == "FFN2" and "FFN3" in role_names:
                current_reference_value = 2 * reference_value
            else:
                current_reference_value = 1 * reference_value

            # Check if matmul global write is within the range of 2-32 * current reference value
            if (
                matmul_global_write < 2 * current_reference_value
                or matmul_global_write > 32 * current_reference_value
            ):
                continue

            # Look for kernel K directly after this matmul with global write == current reference value
            k_idx = matmul_idx + 1
            if k_idx >= len(working):
                continue

            # Skip if this kernel already has a role assigned
            if working.at[k_idx, "role"] != "other":
                continue

            k_global_write = working.at[k_idx, "global_op_st_lookup_miss_bytes"]
            k_global_read = working.at[k_idx, "global_op_ld_lookup_miss_bytes"]

            # Check constraints:
            # 1. For FFN1 and FFN2 (when FFN3 exists), since the ratio between d_ffn and d is unknown:
            #       k global write >= current reference value is a loose constraint
            #    For others: k global write matches current reference value (with small tolerance)
            # 2. Always: matmul global write == k global read (with small tolerance)
            # 3. Kernel K global read > 2 * kernel K global write
            #    This ensures the kernel reads significantly more data than it writes,
            #    which is characteristic of a reduction operation in split-K pattern
            is_ffn_special_case = (role == "FFN1") or (
                role == "FFN2" and "FFN3" in role_names
            )

            if is_ffn_special_case:
                k_gw_constraint_met = k_global_write >= current_reference_value
            else:
                k_gw_constraint_met = np.isclose(
                    k_global_write, current_reference_value, rtol=0.1
                )

            if (
                k_gw_constraint_met
                and np.isclose(matmul_global_write, k_global_read, rtol=0.1)
                and k_global_read > 2 * k_global_write
            ):
                working.at[k_idx, "role"] = "splitKreduce"
                kernel_name = working.at[k_idx, "Kernel Name"]
                splitkreduce_assigned.append(
                    (k_idx, role, kernel_name, k_global_write, k_global_read)
                )
                matmul_gw_ratio = matmul_global_write / current_reference_value
                k_gw_ratio = k_global_write / current_reference_value
                k_gr_ratio = k_global_read / current_reference_value
                logger.info(
                    f"splitKreduce assigned: kernel {k_idx} after {role} matmul "
                    f"(matmul_gw={matmul_global_write:.0f}, k_gw={k_global_write:.0f}, k_gr={k_global_read:.0f}, ref={current_reference_value:.0f}, "
                    f"matmul_ratio={matmul_gw_ratio:.2f}x, k_gw_ratio={k_gw_ratio:.2f}x, k_gr_ratio={k_gr_ratio:.2f}x)"
                )


def run_two_phase_analysis(
    df: pd.DataFrame,
    cfg: AnalyzerConfig,
    *,
    head_rows: int = 250,
    T_max: int = 64,
    eps_factor: float = 0.93,
) -> Tuple[int, int, int, int, int, pd.DataFrame]:
    """
      phase-1: estimate T1 on head, then on full df compute start1 + layer_count (kept forever), peel to end1
      phase-2: estimate T2 on tail-head, then return tail for repeated peeling with fixed layer_count
    """
    head_rows = min(head_rows, len(df))
    df_head = df.iloc[:head_rows].copy().reset_index(drop=True)

    a_head = Analyzer.from_df(df_head, cfg)
    T1 = a_head.detect_layer_period_T(T_max=T_max)

    a_full = Analyzer.from_df(df, cfg)
    start1 = a_full.find_start_point(df, T1, max_search_s=256)
    layer_count = a_full.compute_layer_count_from_start(
        df, start=start1, T=T1, eps_factor=eps_factor
    )
    end1 = start1 + layer_count * T1

    df_tail = df.iloc[end1:].copy().reset_index(drop=True)
    if len(df_tail) < 2:
        return T1, start1, layer_count, end1, T1, df_tail

    df_tail_head = (
        df_tail.iloc[: min(head_rows, len(df_tail))].copy().reset_index(drop=True)
    )
    a_tail_head = Analyzer.from_df(df_tail_head, cfg)
    T2 = a_tail_head.detect_layer_period_T(T_max=T_max)

    return T1, start1, layer_count, end1, T2, df_tail


def main():
    parser = argparse.ArgumentParser(description="PMC Analyzer")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="processed.csv", required=True)
    # If not provided, we average across all detected token cycles.
    parser.add_argument("-t", "--token", type=int, default=None)
    parser.add_argument("--family", default="gpt2", choices=["gpt2", "llama", "qwen"])
    args = parser.parse_args()

    cfg = AnalyzerConfig.for_family(args.family)
    analyzer = Analyzer(df_path=args.input, config=cfg)
    df = analyzer.data

    # Phase 1: Get T1, start_point, layer_num, erase to end
    T1, start1, layer_count, end1, T2, df_tail = run_two_phase_analysis(
        df, cfg, head_rows=250, T_max=64
    )

    # Phase 2: Get T2, start_point, erase to end, repeat with T2 and layer_num from phase 1
    token_cycles = analyzer.peel_token_cycles(
        df_tail, T=T2, layer_count=layer_count, max_search_s=256, absolute_offset=end1
    )

    analyzer.layer_T = T2
    analyzer.last_token_cycles = token_cycles
    logger.info(f"Phase 1: T1={T1}, start={start1}, layers={layer_count}, end={end1}")
    logger.info(f"Phase 2: T2={T2}, tokens={len(token_cycles)}")

    if not token_cycles:
        logger.warning("No token cycles found")
        return

    if args.token is None:
        token_cycle_dfs = []
        for tok_s, tok_e in token_cycles:
            token_cycle_dfs.append(df.iloc[tok_s:tok_e].reset_index(drop=True))
        concat_df = pd.concat(token_cycle_dfs, ignore_index=True)

        token_avg_cycle = analyzer._average_token_cycle(
            concat_df, T2, layer_count
        )
        layer_avg = analyzer._average_layers_by_period(
            token_avg_cycle, T2, layer_count
        )
        roles_df = analyzer.assign_roles(layer_avg)
        saved_desc = f"token-averaged ({len(token_cycles)} tokens)"
    else:
        idx = max(0, min(args.token, len(token_cycles) - 1))
        tok_s, tok_e = token_cycles[idx]
        slice_df = df.iloc[tok_s:tok_e].reset_index(drop=True)
        layer_avg = analyzer._average_layers_by_period(slice_df, T2, layer_count)
        roles_df = analyzer.assign_roles(layer_avg)
        saved_desc = f"token #{idx}"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    roles_df.to_csv(args.output, index=False)
    logger.success(f"Saved {saved_desc} → {args.output}")


if __name__ == "__main__":
    main()
