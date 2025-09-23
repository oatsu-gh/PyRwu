from warnings import warn

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
        g_value = params.flags.params['g'].value
        if g_value == 0:
            return params.sp

        g_value = np.clip(g_value, -99.9, 99.9)  # ゼロ除算防止のためクリッピング
        ratio: float = 1 - g_value / 100

        framerate = params.framerate
        fft_size: int = params.sp.shape[1] - 1
        sp: np.ndarray = params.sp.copy()
        new_sp: np.ndarray = params.sp.copy()

        # original frequency axis. shape: (fft_size // 2,)
        x_freq = np.arange(fft_size // 2) / fft_size * framerate
        # new frequency axis. shape: (fft_size // 2,)
        xi_freq = x_freq / ratio
        # original spectrogram in log-scale. shape: (n_frames, fft_size // 2)
        y_log_sp = np.log(sp[:][: fft_size // 2])
        # interpolated spectrogram in log-scale. shape: (n_frames, fft_size // 2)
        yi_log_sp = np.empty_like(y_log_sp)

        print('fft_size:', fft_size)
        print('ratio:', ratio)
        print('framerate:', framerate)
        print('params.f0.shape:', params.f0.shape)
        print('x_freq.shape:', x_freq.shape)
        print('xi_freq.shape:', xi_freq.shape)
        print('y_log_sp.shape:', y_log_sp.shape)
        print('yi_log_sp.shape:', yi_log_sp.shape)
        print('fft_size // 2:', fft_size // 2)
        # check shapes
        assert x_freq.shape == xi_freq.shape == (fft_size // 2,)

        # ループ無しで処理したい
        for i in range(params.f0.shape[0]):
            y_slice = y_log_sp[i][: fft_size // 2]
            # 線形内挿
            yi_slice = np.interp(xi_freq, x_freq, y_slice)
            # オーバーフロー回避のためクリッピング (float64 の exp に渡す絶対値上限は約709.7827)
            if np.any(np.abs(yi_slice) > 709):
                msg = f'Large spectrum values detected (max: {np.max(np.abs(yi_slice)):.2f}), clipping to ±709'
                warn(msg, stacklevel=2)
                yi_slice = np.clip(yi_slice, -709, 709)

            # 対数を元に戻す
            new_sp[i][: fft_size // 2] = np.exp(yi_slice)

            # ratio が 1.0 未満(男声側へのシフト)の場合は、のこりの周波数成分は元の sp の直前の値で埋める
            if ratio < 1.0:
                j = int(fft_size / 2 * ratio)
                new_sp[i][j : fft_size // 2 + 1] = sp[i][int(fft_size / 2 * ratio) - 1]

        return new_sp

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
        # numpy.interpを使用した1次元線形補間
        yi[:] = np.interp(xi[:xi_length], x[:x_length], y[:x_length])
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
