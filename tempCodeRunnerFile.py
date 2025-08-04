einsum('btm,btd->bmd', kP, v)
        S_prefix_sum = torch.cumsum(S, dim=1)
        N = torch.einsum('btm,bmd->btd', qP, S_prefix_sum)

        # Denominator
        k_sum_prefix = torch.cumsum(kP, dim=1)
        D = torch.einsum