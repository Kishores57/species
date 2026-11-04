import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    Image as RLImage,
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0F766E"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "TREEVISION AI")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(558, 752, "Model Technical Specifications & Training Details")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 744, 558, 744)

        # Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 34, footer_text)
        self.drawString(54, 34, "TreeVisionAI — Deep Learning Species Classification Pipeline")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_pdf(filename="TreeVisionAI_Technical_Specifications.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#0F766E")     # Deep Forest Teal
    SECONDARY = colors.HexColor("#1E293B")   # Slate Navy
    TEXT_DARK = colors.HexColor("#334155")   # Charcoal Body Text
    ACCENT_BG = colors.HexColor("#F8FAFC")   # Light Gray Fill
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=3,
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=9,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15.5,
        textColor=SECONDARY,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=12.5,
        textColor=TEXT_DARK,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=12,
        textColor=TEXT_DARK,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=11,
        textColor=colors.white,
        alignment=0,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.8,
        textColor=TEXT_DARK,
        alignment=0,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.8,
        textColor=SECONDARY,
        alignment=0,
    )

    story = []

    # ==========================================
    # PAGE 1: Overview, Architecture, Training Strategy
    # ==========================================
    story.append(Paragraph("TreeVisionAI: Model Technical Specifications", title_style))
    story.append(Paragraph("Complete technical dossier covering model architecture, two-phase training hyperparameters, and optimization.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=0, spaceAfter=8))

    meta_data = [
        [
            Paragraph("<b>Base Architecture:</b> EfficientNet-B0 (ImageNet-1K V1)", table_cell_style),
            Paragraph("<b>Target Classes:</b> 7 Native Tree Species", table_cell_style),
        ],
        [
            Paragraph("<b>Final Val Accuracy:</b> 92.35% (Peak: 92.89%)", table_cell_style),
            Paragraph("<b>Input Resolution:</b> 224 × 224 × 3 px", table_cell_style),
        ],
        [
            Paragraph("<b>Training Strategy:</b> 2-Phase Transfer Learning (40 Epochs)", table_cell_style),
            Paragraph("<b>Saved Checkpoint:</b> models/best_model.pth", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT_BG),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Section 1: Model Architecture
    story.append(Paragraph("1. Model Architecture & Classification Head", h1_style))
    story.append(Paragraph("The system is engineered using <b>EfficientNet-B0</b> with pretrained weights (<code>EfficientNet_B0_Weights.IMAGENET1K_V1</code>). The original 1000-class linear layer was removed and replaced with a custom lightweight head designed for 7-class botanical taxonomy:", body_style))
    
    story.append(Paragraph("• <b>Feature Extractor (Backbone):</b> Inverted residual MBConv blocks optimized via Squeeze-and-Excitation attention.", bullet_style))
    story.append(Paragraph("• <b>Spatial Pooling:</b> Global Adaptive Average Pooling (<code>nn.AdaptiveAvgPool2d(1)</code>) compressing spatial features to 1280 dims.", bullet_style))
    story.append(Paragraph("• <b>Dropout Regularization:</b> Inverted dropout with <code>p=0.2</code> (<code>nn.Dropout(p=0.2, inplace=True)</code>) to suppress co-adaptation.", bullet_style))
    story.append(Paragraph("• <b>Classification Projection:</b> Dense layer <code>nn.Linear(in_features=1280, out_features=7)</code>.", bullet_style))
    story.append(Paragraph("• <b>Target Species Classes:</b> 1. <b>Ashoka</b> (<i>Polyalthia longifolia</i>), 2. <b>Banyan</b> (<i>Ficus benghalensis</i>), 3. <b>Mango</b> (<i>Mangifera indica</i>), 4. <b>Neem</b> (<i>Azadirachta indica</i>), 5. <b>Peepal</b> (<i>Ficus religiosa</i>), 6. <b>Rain Tree</b> (<i>Samanea saman</i>), 7. <b>Rubber Tree</b> (<i>Ficus elastica</i>).", bullet_style))

    story.append(Spacer(1, 8))

    # Section 2: Two-Phase Training Strategy
    story.append(Paragraph("2. Two-Phase Transfer Learning Hyperparameters", h1_style))
    story.append(Paragraph("To protect pretrained ImageNet representations while adapting to tree foliage, a two-phase progressive training pipeline was employed:", body_style))

    train_data = [
        [
            Paragraph("Hyperparameter / Setting", table_header_style),
            Paragraph("Phase 1: Classifier Head Warmup", table_header_style),
            Paragraph("Phase 2: Global Fine-Tuning", table_header_style)
        ],
        [
            Paragraph("Trainable Parameters", table_cell_bold),
            Paragraph("Classifier Head only (Backbone Frozen)", table_cell_style),
            Paragraph("All Layers (Backbone Unfrozen)", table_cell_style)
        ],
        [
            Paragraph("Epoch Count", table_cell_bold),
            Paragraph("10 Epochs", table_cell_style),
            Paragraph("30 Epochs (Total: 40 Global Epochs)", table_cell_style)
        ],
        [
            Paragraph("Optimizer & Weight Decay", table_cell_bold),
            Paragraph("AdamW (Weight Decay: 1e-4)", table_cell_style),
            Paragraph("AdamW (Weight Decay: 1e-4)", table_cell_style)
        ],
        [
            Paragraph("Base Learning Rate (LR)", table_cell_bold),
            Paragraph("1e-3 (0.001)", table_cell_style),
            Paragraph("Initial 1e-4 (0.0001) scheduled down to 1e-6", table_cell_style)
        ],
        [
            Paragraph("Learning Rate Scheduler", table_cell_bold),
            Paragraph("Constant LR", table_cell_style),
            Paragraph("CosineAnnealingLR (T_max=30, eta_min=1e-6)", table_cell_style)
        ],
        [
            Paragraph("Batch Size & DataLoader Workers", table_cell_bold),
            Paragraph("32 (Workers: 4)", table_cell_style),
            Paragraph("32 (Workers: 4)", table_cell_style)
        ],
        [
            Paragraph("Early Stopping Mechanism", table_cell_bold),
            Paragraph("Disabled (Full Warmup)", table_cell_style),
            Paragraph("Patience: 7 epochs, min_delta: 0.001", table_cell_style)
        ],
    ]
    t_train = Table(train_data, colWidths=[130, 184, 190])
    t_train.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ACCENT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_train)

    # Clean PageBreak to Page 2
    story.append(PageBreak())

    # ==========================================
    # PAGE 2: Techniques, Preprocessing, Tech Stack, Milestone Summary
    # ==========================================
    story.append(Paragraph("3. Optimization & Training Techniques", h1_style))
    story.append(Paragraph("• <b>Label Smoothing (0.1):</b> Configured via <code>nn.CrossEntropyLoss(label_smoothing=0.1)</code>. Softens standard one-hot encodings into distributed probabilities, dampening overconfidence on hard visual boundaries.", bullet_style))
    story.append(Paragraph("• <b>Automatic Mixed Precision (AMP):</b> Utilized <code>torch.amp.autocast('cuda')</code> coupled with <code>GradScaler</code> for FP16 tensor operations, reducing memory footprint on GPU while preserving numerical stability.", bullet_style))
    story.append(Paragraph("• <b>Gradient Norm Clipping:</b> Bounded backpropagated gradients via <code>nn.utils.clip_grad_norm_(max_norm=1.0)</code> to avert gradient explosion during Phase 2 unfreezing.", bullet_style))
    story.append(Paragraph("• <b>Class Imbalance Handling:</b> Integrated PyTorch <code>WeightedRandomSampler</code> with inverse-frequency weights (<i>w<sub>c</sub> = 1 / N<sub>c</sub></i>) to present a balanced class distribution per training batch.", bullet_style))
    story.append(Paragraph("• <b>Deterministic Splitting:</b> Stratified split into 80% Train, 10% Validation, and 10% Test with <code>seed=42</code>.", bullet_style))

    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Data Augmentation & Preprocessing Pipeline", h1_style))
    story.append(Paragraph("• <b>In-Flight Augmentation (Train Loader):</b> RandomResizedCrop (scale 0.7-1.0, aspect ratio 0.8-1.2), RandomHorizontalFlip (p=0.5), RandomRotation (±15°), RandomPerspective (distortion 0.2, p=0.3), ColorJitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05), and RandomAffine (translation ±10%, scaling 0.9-1.1).", bullet_style))
    story.append(Paragraph("• <b>Inference & Validation Transform:</b> Deterministic Resize to 256px followed by a 224×224 CenterCrop and channel-wise ImageNet standard normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).", bullet_style))
    story.append(Paragraph("• <b>Dataset Cleaning Pipeline (OpenCV & ImageHash):</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Laplacian Blur Filter:</i> Discarded images with Laplacian variance < 100.0.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Luminance Outlier Filter:</i> Rejected under-exposed (< 30) and over-exposed (> 240) images.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Multi-Stage Deduplication:</i> Exact match removal with SHA-256; near-duplicate elimination via pHash (distance ≤ 5).", bullet_style))

    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Core Software Libraries & Stack", h1_style))
    tech_data = [
        [
            Paragraph("Domain / Category", table_header_style),
            Paragraph("Core Libraries", table_header_style),
            Paragraph("Role in System", table_header_style)
        ],
        [
            Paragraph("Deep Learning", table_cell_bold),
            Paragraph("torch, torchvision", table_cell_style),
            Paragraph("Neural model architecture, transforms, DataLoader, AMP, Autograd", table_cell_style)
        ],
        [
            Paragraph("Computer Vision", table_cell_bold),
            Paragraph("opencv-python, pillow, imagehash", table_cell_style),
            Paragraph("Laplacian variance blur, luminance filtering, perceptual hashing", table_cell_style)
        ],
        [
            Paragraph("Data & Statistics", table_cell_bold),
            Paragraph("numpy, pandas, scikit-learn", table_cell_style),
            Paragraph("Data wrangling, confusion matrix, classification metrics", table_cell_style)
        ],
        [
            Paragraph("Visualization", table_cell_bold),
            Paragraph("matplotlib, seaborn", table_cell_style),
            Paragraph("Training history plots, confusion matrix heatmaps", table_cell_style)
        ],
        [
            Paragraph("Deployment", table_cell_bold),
            Paragraph("fastapi, uvicorn", table_cell_style),
            Paragraph("Asynchronous REST API server for single-image leaf predictions", table_cell_style)
        ],
    ]
    t_tech = Table(tech_data, colWidths=[110, 160, 234])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ACCENT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Performance Milestone Summary", h1_style))
    perf_data = [
        [
            Paragraph("<b>Metric</b>", table_header_style),
            Paragraph("<b>Phase 1 Head (Epoch 10)</b>", table_header_style),
            Paragraph("<b>Phase 2 Best (Epoch 38)</b>", table_header_style),
            Paragraph("<b>Final Epoch (Epoch 40)</b>", table_header_style)
        ],
        [
            Paragraph("Training Accuracy", table_cell_bold),
            Paragraph("59.07%", table_cell_style),
            Paragraph("98.46%", table_cell_style),
            Paragraph("98.46%", table_cell_style)
        ],
        [
            Paragraph("Validation Accuracy", table_cell_bold),
            Paragraph("65.44%", table_cell_style),
            Paragraph("<b>92.89%</b> (Peak)", table_cell_style),
            Paragraph("<b>92.35%</b>", table_cell_style)
        ],
        [
            Paragraph("Training Loss", table_cell_bold),
            Paragraph("1.3245", table_cell_style),
            Paragraph("0.5206", table_cell_style),
            Paragraph("0.5213", table_cell_style)
        ],
        [
            Paragraph("Validation Loss", table_cell_bold),
            Paragraph("1.2126", table_cell_style),
            Paragraph("0.6388", table_cell_style),
            Paragraph("0.6333", table_cell_style)
        ],
    ]
    t_perf = Table(perf_data, colWidths=[120, 128, 128, 128])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ACCENT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_perf)

    # Clean PageBreak to Page 3
    story.append(PageBreak())

    # ==========================================
    # PAGE 3: Visual Performance Artifacts
    # ==========================================
    story.append(Paragraph("7. Model Evaluation & Performance Visualizations", h1_style))
    story.append(Paragraph("Progression trajectories and categorical confusion matrix across test/validation partitions:", body_style))
    story.append(Spacer(1, 4))

    acc_path = os.path.join("outputs", "accuracy_curve.png")
    loss_path = os.path.join("outputs", "loss_curve.png")
    cm_path = os.path.join("outputs", "confusion_matrix.png")

    if os.path.exists(acc_path) and os.path.exists(loss_path):
        curves_table = Table([
            [
                RLImage(acc_path, width=244, height=135),
                RLImage(loss_path, width=244, height=135)
            ],
            [
                Paragraph("<b>Figure 1:</b> Training & Validation Accuracy across 40 Epochs", table_cell_style),
                Paragraph("<b>Figure 2:</b> Training & Validation Loss across 40 Epochs", table_cell_style)
            ]
        ], colWidths=[252, 252])
        curves_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(curves_table)
        story.append(Spacer(1, 10))

    if os.path.exists(cm_path):
        cm_table = Table([
            [
                RLImage(cm_path, width=310, height=230)
            ],
            [
                Paragraph("<b>Figure 3:</b> Multi-Class Confusion Matrix Heatmap (7 Target Species)", table_cell_style)
            ]
        ], colWidths=[504])
        cm_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(cm_table)
        story.append(Spacer(1, 8))

    # Concluding take-away card
    summary_box = [
        [
            Paragraph(
                "<b>Key Takeaways:</b> The model demonstrates superior discriminative capacity with high true-positive density along the primary diagonal. Minor confusion appears between morphologically similar species (e.g., young Banyan vs. Rubber tree leaves), while distinctive leaf structures (Ashoka, Neem, Rain Tree) achieve >95% per-class precision.",
                body_style
            )
        ]
    ]
    t_summary = Table(summary_box, colWidths=[504])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT_BG),
        ('BOX', (0, 0), (-1, -1), 1, PRIMARY),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_summary)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully: {filename}")

if __name__ == "__main__":
    build_pdf()
