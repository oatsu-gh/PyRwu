import numpy as np
from scipy.interpolate import make_interp_spline

from .base import WorldEffectBase


def shift_spectrum(
    sp: np.ndarray,
    framerate: int,
    ratio: float,
    scale_kind: str = 'mel',
    interp_kind: str = 'cubic',
) -> np.ndarray:
    """スペクトルを周波数軸方向にシフトする (フォルマントシフト)

    Args:
        sp: スペクトル (n_frames, n_fft // 2 + 1)
        ratio: シフト比率 (>1.0 で高周波側へシフト、<1.0 で低周波側へシフト)

    Returns:
        シフト後のスペクトル (n_frames, n_fft // 2 + 1)
    """
    # シフト比率が1.0の場合はそのまま返す
    if ratio == 1.0:
        return sp

    # 入力スペクトルの形状を取得
    _n_frames, n_bins = sp.shape
    _fft_size = (n_bins - 1) * 2

    # 周波数ビン数のチェック
    if n_bins < 2:
        msg = f'Not enough frequency bins: {n_bins}'
        raise ValueError(msg)

    # 補間の次数を決める
    if interp_kind == 'linear':
        k = 1
    elif interp_kind == 'cubic':
        k = 3
    elif interp_kind == 'quintic':
        k = 5
    else:
        msg = f'Unknown interpolation kind: {interp_kind}'
        raise ValueError(msg)
    # 補間次数をデータ点数-1以下に制限
    k = min(k, n_bins - 1)

    # スケール変換関数
    def hz_to_scale(frq: np.ndarray) -> np.ndarray:
        """周波数 (Hz) からスケールへの変換"""
        if scale_kind == 'mel':
            return 2595 * np.log10(1 + frq / 700)
        if scale_kind == 'log':
            return np.log1p(frq)
        if scale_kind == 'linear':
            return frq
        msg = f'Unknown spectrum scale: {scale_kind}'
        raise ValueError(msg)

    def scale_to_hz(scl: np.ndarray) -> np.ndarray:
        """スケールから周波数 (Hz) への変換"""
        if scale_kind == 'mel':
            return 700 * (10 ** (scl / 2595) - 1)
        if scale_kind == 'log':
            return np.expm1(scl)
        if scale_kind == 'linear':
            return scl
        msg = f'Unknown spectrum scale: {scale_kind}'
        raise ValueError(msg)

    # 周波数軸 (Hz)
    x_orig = np.linspace(0, framerate / 2, n_bins)
    # 軸スケールを変換
    x_scaled_freqs = hz_to_scale(x_orig)
    xi_scaled_freqs = x_scaled_freqs / ratio
    # 範囲外の値をクリップ。外挿防止のため。NOTE: クリッピングこの範囲でいいか検討。
    xi_scaled_freqs = np.clip(xi_scaled_freqs, x_scaled_freqs[0], x_scaled_freqs[-1])
    # 周波数軸のスケールをを元に戻す
    xi_freqs = scale_to_hz(xi_scaled_freqs)

    # 出力配列
    new_sp = np.zeros_like(sp)

    # 全フレームを一度に補間 (ベクトル化)
    ## フレームごとにスプライン補間する場合のコード---------
    # for i in range(sp.shape[0]):
    #     spline = make_interp_spline(xi_freqs, sp[i], k=k, axis=0)
    #     new_sp[i] = spline(xi_scaled_freqs)
    ## 全フレームを一度に補間することで高速化----------------
    spline = make_interp_spline(x_orig, sp.T, k=k, axis=0)
    new_sp = spline(xi_freqs).T.copy()  # .T はビューなのでWORLDでエラーになるため copy
    new_sp = np.clip(
        new_sp,
        np.finfo(new_sp.dtype).tiny,
        None,
    )  # 負の値をほぼ0にクリップ
    return new_sp


class GFlag(WorldEffectBase):
    @staticmethod
    def apply(params, scale_kind='mel', interp_kind='cubic') -> np.ndarray:
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
        ratio = 1 - g_value / 100

        sp = params.sp.copy()
        framerate = params.framerate
        new_sp = shift_spectrum(
            sp,
            framerate,
            ratio,
            scale_kind=scale_kind,
            interp_kind=interp_kind,
        )

        if not np.all(np.isfinite(new_sp)):
            msg = 'Non-finite values found in shifted spectrum.'
            raise ValueError(msg)
        if new_sp.shape != params.sp.shape:
            msg = f'Original sp shape ({params.sp.shape}) and shifted sp shapes ({new_sp.shape}) do not match.'
            raise ValueError(msg)

        return new_sp
