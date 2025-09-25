import numpy as np

from .base import WorldEffectBase


class LargeBFlag(WorldEffectBase):
    @staticmethod
    def apply(params) -> np.ndarray:
        """
        | 息成分の強さ(ブレシネス)。大きいほど息っぽい
        | 0～49ではB0の時非周期性指標が全て0になるように乗算します。
        | 51～100ではB100の時、1000Hz～5000Hz帯の非周期性指標が全て1になるように加算します。

        Parameters
        ----------
        params: resamp.Resamp

            伸縮機の各パラメータ

        Returns
        -------
        new_values: np.ndarray of float64

            | 処理後の値

        """
        if params.flags.params['B'].value == params.flags.params['B'].default_value:
            return params.ap

        new_ap: np.ndarray = params.ap.copy()
        if params.flags.params['B'].value < params.flags.params['B'].default_value:
            new_ap = (
                new_ap
                * params.flags.params['B'].value
                / (
                    params.flags.params['B'].default_value
                    - params.flags.params['B'].min
                )
            )
        else:
            effect: np.ndarray = np.ones_like(new_ap) - new_ap
            mask: np.ndarray = np.zeros_like(new_ap)
            # 1000Hz以下をマスク
            mask_len: int = int(1000 * (new_ap.shape[1] - 1) / params.framerate)
            effect[:, :mask_len] = mask[:, :mask_len]
            mask_len: int = int(5000 * (new_ap.shape[1] - 1) / params.framerate)
            effect[:, mask_len:] = mask[:, mask_len:]
            new_ap = new_ap + effect * (
                params.flags.params['B'].value - params.flags.params['B'].default_value
            ) / (params.flags.params['B'].max - params.flags.params['B'].default_value)

        # 0.001～1 の範囲にクリップ (wav生成時にエラーになるのを防ぐため)
        new_ap = np.clip(new_ap, 0.001, 1.0)
        return new_ap
