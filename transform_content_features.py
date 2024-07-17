import torch


def ReTrans(source_feats, padded_target_len):
        """
        Resolution Transformation for mismatched frames alginment.

        TODO: Merge the offline resolution_transformation into one

        args:
            source_feats: Tensor, (B, padded_source_len, D)
            padded_target_len: int, the maximum target length in a batch
        return:
            mapped_feature: Tensor, (B, padded_target_len, D)
        """
        source_hop = self.source_hop
        target_hop = self.target_hop

        # (B, padded_source_len, D)
        B, padded_source_len, D = source_feats.shape

        # select the valid content from padded feature
        source_len = min(
            padded_target_len * target_hop // source_hop + 1, padded_source_len
        )

        # const ~= padded_target_len * target_hop (padded wav's duration)
        const = source_len * source_hop // target_hop * target_hop

        # (B, padded_source_len, D) -> (B, padded_source_len * source_hop, D) -> (B, const, D)
        up_sampling_feats = torch.repeat_interleave(source_feats, source_hop, dim=1)[
            :, :const
        ]
        # (B, const, D) -> (B, const/target_hop, target_hop, D) -> (B, const/target_hop, D)
        down_sampling_feats = torch.mean(
            up_sampling_feats.reshape(B, -1, target_hop, D), dim=2
        )

        err = abs(padded_target_len - down_sampling_feats.shape[1])
        if err > 8:
            self.log_for_ReTrans(err)

        if down_sampling_feats.shape[1] < padded_target_len:
            # (B, 1, D) -> (B, err, D)
            end = down_sampling_feats[:, -1, :][:, None, :].repeat_interleave(
                err, dim=1
            )
            # -> (B, padded_target_len, D)
            down_sampling_feats = torch.cat([down_sampling_feats, end], dim=1)

        # (B, padded_target_len, D)
        mapped_feature = down_sampling_feats[:, :padded_target_len]

        return mapped_feature