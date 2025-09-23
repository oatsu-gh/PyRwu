import numpy as np
from .base import WorldEffectBase


class GFlag(WorldEffectBase):
    @staticmethod
    def apply(params) -> np.ndarray:
        """
        | 疑似ジェンダー値
        | 負の数で女声化・若年化
        | 正の数で男声化・大人化します。

        Parameters
        ----------
        params: resamp.Resamp

            伸縮機の各パラメータ

        Returns
        -------
        new_values: np.ndarray of float64

            | 処理後の値

        """
        if params.flags.params["g"].value == 0:
            return params.sp

        ratio: float = 1 - params.flags.params["g"].value / 100
        fft_size: int = params.sp.shape[1] - 1
        freq_axis1: np.ndarray = np.ndarray(fft_size)
        freq_axis2: np.ndarray = np.ndarray(fft_size)
        spectrum1: np.ndarray = np.ndarray(fft_size)
        spectrum2: np.ndarray = np.ndarray(fft_size)
        sp: np.ndarray = params.sp.copy()
        for i in range(int(fft_size / 2)):
            freq_axis1[i] = ratio * i / fft_size * params.framerate
            freq_axis2[i] = i / fft_size * params.framerate

        for i in range(params.f0.shape[0]):
            for j in range(int(fft_size / 2)):
                # Prevent log of zero or negative values by using a small positive minimum
                spectrum1[j] = np.log(np.maximum(sp[i][j], 1e-30))
            spectrum2 = GFlag._interp1(
                freq_axis1,
                spectrum1,
                int(fft_size / 2) + 1,
                freq_axis2,
                int(fft_size / 2) + 1,
                spectrum2,
            )
            for j in range(int(fft_size / 2)):
                sp[i][j] = np.exp(spectrum2[j])
            if ratio >= 1.0:
                continue
            j = int(fft_size / 2 * ratio)
            while j <= fft_size / 2:
                sp[i][j] = sp[i][int(fft_size / 2 * ratio) - 1]
                j = j + 1
        
        # Final safety check: ensure no zeros remain that could cause issues downstream
        sp = np.maximum(sp, 1e-30)
        return sp

    @staticmethod
    def _interp1(x, y, x_length, xi, xi_length, yi):
        h = np.zeros(x_length)
        k = np.zeros(xi_length, dtype=np.int32)

        for i in range(x_length - 1):
            h[i] = x[i + 1] - x[i]
        k = GFlag._histc(x, x_length, xi, xi_length, k)
        for i in range(xi_length):
            # Ensure array bounds are respected
            k_idx = max(0, min(k[i] - 1, x_length - 1))
            k_next = max(0, min(k[i], x_length - 1))
            
            # Prevent division by zero by checking if h[k_idx] is effectively zero
            if abs(h[k_idx]) < 1e-30:
                # If the interval is too small, use the nearest neighbor value
                yi[i] = y[k_idx]
            else:
                s = (xi[i] - x[k_idx]) / h[k_idx]
                yi[i] = y[k_idx] + s * (y[k_next] - y[k_idx])
        return yi

    @staticmethod
    def _histc(x, x_length, edges, edges_length, index):
        count = 1
        for i in range(edges_length):
            index[i] = 1
            if edges[i] >= x[0]:
                break
        while i < edges_length:
            # Add bounds checking to prevent array access errors
            if count >= x_length:
                break
            if edges[i] < x[count]:
                index[i] = count
            else:
                i = i - 1
                if i >= 0:  # Ensure i doesn't go negative
                    index[i] = count
                count = count + 1
            if count == x_length:
                break
            i = i + 1
        count = count - 1
        i = i + 1
        while i < edges_length:
            index[i] = count
            i = i + 1
        return index
