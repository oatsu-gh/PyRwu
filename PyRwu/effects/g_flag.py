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
        if params.flags.params['g'].value == 0:
            return params.sp

        framerate = params.framerate

        ratio: float = 1 - params.flags.params['g'].value / 100
        fft_size: int = params.sp.shape[1] - 1
        freq_axis1: np.ndarray = np.arange(fft_size // 2) * ratio / fft_size * framerate
        freq_axis2: np.ndarray = np.arange(fft_size // 2) / fft_size * framerate
        spectrum1: np.ndarray = np.ndarray(fft_size)
        spectrum2: np.ndarray = np.ndarray(fft_size)
        sp: np.ndarray = params.sp.copy()

        for i in range(params.f0.shape[0]):
            spectrum1[: fft_size // 2] = np.log(sp[i][: fft_size // 2])
            spectrum2 = GFlag._interp1(
                freq_axis1,
                spectrum1,
                fft_size // 2 + 1,
                freq_axis2,
                fft_size // 2 + 1,
                spectrum2,
            )
            spectrum_slice = spectrum2[: fft_size // 2]
            if np.any(np.abs(spectrum_slice) > 709):
                warn(
                    f'Large spectrum values detected (max: {np.max(np.abs(spectrum_slice)):.2f}), clipping to ±709',
                    stacklevel=2,
                )
                spectrum_slice = np.clip(spectrum_slice, -709, 709)
            sp[i][: fft_size // 2] = np.exp(spectrum_slice)
            if ratio >= 1.0:
                continue
            j = int(fft_size / 2 * ratio)
            sp[i][j : fft_size // 2 + 1] = sp[i][int(fft_size / 2 * ratio) - 1]
        return sp

    @staticmethod
    def _interp1(x, y, x_length, xi, xi_length, yi):
        """1次元線形補間

        Args:
            x: 既知のx座標配列（昇順ソート済み）
            y: 既知のy値配列
            x_length: xの長さ
            xi: 補間したいx座標配列
            xi_length: xiの長さ
            yi: 補間結果を格納する配列
        """
        h = np.zeros(x_length)
        k = np.zeros(xi_length, dtype=np.int32)

        # 各区間の幅を計算
        for i in range(x_length - 1):
            h[i] = x[i + 1] - x[i]
        # 補間点がどの区間に属するかを特定
        k = GFlag._histc(x, x_length, xi, xi_length, k)
        # 線形補間を実行
        for i in range(xi_length):
            s = (xi[i] - x[k[i] - 1]) / h[k[i] - 1]
            yi[i] = y[k[i] - 1] + s * (y[k[i]] - y[k[i] - 1])
        return yi

    @staticmethod
    def _histc(x, x_length, edges, edges_length, index):
        count = 1
        for i in range(edges_length):
            index[i] = 1
            if edges[i] >= x[0]:
                break
        while i < edges_length:
            if edges[i] < x[count]:
                index[i] = count
            else:
                i = i - 1
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
