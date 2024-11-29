from torch import nn


# Custom header
class HeaderBlock(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout_prob:float):
        super(HeaderBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.gelu(x)
        return self.dropout(x)


def custom_binary_header(x:int)->nn.Module:
    """
    Pre-Activation Batch Normalization
    - 깊은 Backbone 모델의 header이므로, Internal Covariate Shift 문제 해결을 위해 사용

    BottleNeck
    - 정보를 확장하여 중요한 정보만 남겨, 계산 효율성을 유지하면서 높은 표현력 제공
    """
    header = nn.Sequential(
        HeaderBlock(input_dim=x, output_dim=1024, dropout_prob=0.1),
        HeaderBlock(input_dim=1024, output_dim=512, dropout_prob=0.3),
        HeaderBlock(input_dim=512, output_dim=128, dropout_prob=0.5),
        nn.Linear(128, 1)
    )
    return header


# 모델 출력 결과에 대한 추가 활성화 함수
EXTRA_ACTIVATION_FN = "sigmoid"