import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Expert(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)


class UnetSharedBackbone(nn.Module):
    def __init__(self, encoder_name="resnet18", pretrained=True):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights="imagenet" if pretrained else None
        )
        self.out_channels = self.encoder.out_channels[-1]

    def forward(self, x):
        return self.encoder(x)  # List of feature maps


from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetMultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder_name="resnet18",
        num_classes=5,
        num_seg_channels=1,
        num_experts=4,
        dropout_p=0.3
    ):
        super().__init__()

        # Shared Backbone
        self.backbone = UnetSharedBackbone(encoder_name)
        self.encoder_out_channels = self.backbone.out_channels

        # === Classification Head ===
        self.gating = GatingNetwork(self.encoder_out_channels, num_experts)
        self.experts = nn.ModuleList([
            Expert(self.encoder_out_channels, self.encoder_out_channels, dropout_p=dropout_p)
            for _ in range(num_experts)
        ])
        self.dropout_cls = nn.Dropout2d(p=dropout_p)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder_out_channels, num_classes)
        )

        # === Segmentation Head ===
        self.decoder = UnetDecoder(
            encoder_channels=self.backbone.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            #use_batchnorm=True,
        )
        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(16, num_seg_channels, kernel_size=1)
        )

    def forward(self, x, task):
        features = self.backbone(x)  # Should return a list of feature maps from encoder

        if task == "classification":
            x_cls = self.dropout_cls(features[-1])
            gate_weights = self.gating(x_cls)
            expert_outputs = [
                expert(x_cls) * gate_weights[:, i].view(-1, 1, 1, 1)
                for i, expert in enumerate(self.experts)
            ]
            combined = sum(expert_outputs)
            return self.classifier(combined)

        elif task == "segmentation":
            x_seg = self.decoder(features)
            return self.segmentation_head(x_seg)

        else:
            raise ValueError("Invalid task: choose 'classification' or 'segmentation'")



class MultiTaskNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=5):
        super().__init__()

        # Shared Encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name, weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels
        bottleneck_dim = encoder_channels[-1]

        # Task Expert 1: Segmentation Decoder
        self.seg_decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            #use_batchnorm=True,
        )
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)  # Binary segmentation

        # Task Expert 2: Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, num_classes)  # e.g., 5-class disease grading
        )

        # Dynamic Router Head
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),     # Reduce to [B, C, 1, 1]
            nn.Flatten(),                     # [B, C]
            nn.Dropout(p=0.2),
            nn.Linear(bottleneck_dim, 2),     # 2 tasks: classification and segmentation
            nn.Softmax(dim=1)                 # Output routing weights
        )

    def forward(self, x):
        # Shared encoder
        features = self.encoder(x)
        bottleneck = features[-1]  # shape: [B, C, H, W]

        # Routing weights
        routing_weights = self.router(bottleneck)  # shape: [B, 2]

        # Task-specific outputs
        seg_out = self.seg_head(self.seg_decoder(features))     # [B, 1, H, W]
        cls_out = self.cls_head(bottleneck)                      # [B, num_classes]

        return {
            'cls_out': cls_out,
            'seg_out': seg_out,
            'routing_weights': routing_weights
        }



'''
class UnetMultiTaskModel(nn.Module):
    def __init__(
        self, 
        encoder_name="resnet18", 
        num_classes=5, 
        num_seg_channels=1, 
        num_experts=4,
        dropout_p=0.3
    ):
        super().__init__()
        self.backbone = UnetSharedBackbone(encoder_name)
        self.encoder_out_channels = self.backbone.out_channels

        # === Classification Head ===
        self.gating = GatingNetwork(self.encoder_out_channels, num_experts)
        self.experts = nn.ModuleList([
            Expert(self.encoder_out_channels, self.encoder_out_channels, dropout_p=dropout_p)
            for _ in range(num_experts)
        ])
        self.dropout_cls = nn.Dropout2d(p=dropout_p)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder_out_channels, num_classes)
        )

        # === Segmentation Head ===
        self.decoder = smp.Unet.decoder.UnetDecoder(
            encoder_channels=self.backbone.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
        )

        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_p),  # Added dropout before final conv
            nn.Conv2d(16, num_seg_channels, kernel_size=1)
        )

    def forward(self, x, task):
        features = self.backbone(x)

        if task == "classification":
            x_cls = self.dropout_cls(features[-1])
            gate_weights = self.gating(x_cls)
            expert_outputs = [
                expert(x_cls) * gate_weights[:, i].view(-1, 1, 1, 1)
                for i, expert in enumerate(self.experts)
            ]
            combined = sum(expert_outputs)
            return self.classifier(combined)

        elif task == "segmentation":
            x_seg = self.decoder(features*)
            return self.segmentation_head(x_seg)

        else:
            raise ValueError("Invalid task: choose 'classification' or 'segmentation'")
'''
