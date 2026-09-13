import base64
import io
import os
from datetime import datetime


def _build_patient_report_reportlab(data, cfg, describe_column, weights, threshold):
    """Return (BytesIO, filename) for a complete patient-specific PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        PageBreak, HRFlowable, LongTable,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    dejavu_regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    dejavu_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        if os.path.exists(dejavu_regular) and os.path.exists(dejavu_bold):
            pdfmetrics.registerFont(TTFont("DashboardSans", dejavu_regular))
            pdfmetrics.registerFont(TTFont("DashboardSansBold", dejavu_bold))
            regular_font = "DashboardSans"
            bold_font = "DashboardSansBold"
    except Exception:
        pass

    pink = colors.HexColor("#E6437A")
    navy = colors.HexColor("#0D1738")
    blue = colors.HexColor("#3E78D7")
    green = colors.HexColor("#2D9F6F")
    amber = colors.HexColor("#D6972F")
    red = colors.HexColor("#C94461")
    pale = colors.HexColor("#F5F7FC")
    line = colors.HexColor("#DDE2EE")
    text_color = colors.HexColor("#20283A")
    muted = colors.HexColor("#667085")

    w_clin, w_2d, w_mil, w_whole = weights

    patient_id = str(data.get("patient_id") or "patient")
    patient_name = str(data.get("patient_name") or "-")
    generated_at = datetime.now().strftime("%d %b %Y %H:%M")
    report_id = f"LC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title=f"Lung Cancer Risk Assessment - {patient_id}",
        author="Lung Cancer AI Analysis System",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RptTitle", fontName=bold_font, fontSize=19, leading=23, textColor=navy, spaceAfter=3 * mm))
    styles.add(ParagraphStyle(name="RptSub", fontName=regular_font, fontSize=8.5, leading=11, textColor=muted, spaceAfter=1.5 * mm))
    styles.add(ParagraphStyle(name="Section", fontName=bold_font, fontSize=12.5, leading=15, textColor=navy, spaceBefore=3 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="BodyRpt", fontName=regular_font, fontSize=8.5, leading=12, textColor=text_color))
    styles.add(ParagraphStyle(name="SmallRpt", fontName=regular_font, fontSize=7.4, leading=10, textColor=muted))
    styles.add(ParagraphStyle(name="SmallBold", fontName=bold_font, fontSize=7.6, leading=10, textColor=text_color))
    styles.add(ParagraphStyle(name="WhiteBold", fontName=bold_font, fontSize=10.5, leading=13, textColor=colors.white, alignment=1))
    styles.add(ParagraphStyle(name="ScoreBig", fontName=bold_font, fontSize=24, leading=28, textColor=colors.white, alignment=1))
    styles.add(ParagraphStyle(name="Appendix", fontName=regular_font, fontSize=6.8, leading=8.4, textColor=text_color))

    def P(value, style="BodyRpt"):
        if value is None or value == "":
            value = "-"
        return Paragraph(str(value), styles[style])

    def pct(prob, digits=1):
        try:
            if prob is None:
                return "Unavailable"
            return f"{float(prob) * 100:.{digits}f}%"
        except Exception:
            return "Unavailable"

    def num(value, digits=4):
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def value_text(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            if v.is_integer():
                return str(int(v))
            return f"{v:.4g}"
        return str(v)

    def table_style(header_bg=navy):
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTNAME", (0, 1), (-1, -1), regular_font),
            ("FONTSIZE", (0, 0), (-1, -1), 7.4),
            ("LEADING", (0, 0), (-1, -1), 9.4),
            ("GRID", (0, 0), (-1, -1), 0.35, line),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])

    def decode_report_image(b64_text, max_w=53 * mm, max_h=47 * mm):
        if not b64_text:
            return None
        try:
            raw = base64.b64decode(b64_text)
            bio = io.BytesIO(raw)
            from PIL import Image as PILImage
            with PILImage.open(bio) as im:
                iw, ih = im.size
            bio.seek(0)
            scale = min(max_w / iw, max_h / ih)
            return Image(bio, width=iw * scale, height=ih * scale)
        except Exception:
            return None

    def header_footer(canvas, report_doc):
        canvas.saveState()
        page_w, _ = A4
        canvas.setStrokeColor(colors.HexColor("#E5E9F2"))
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, 12 * mm, page_w - 15 * mm, 12 * mm)
        canvas.setFillColor(muted)
        canvas.setFont(regular_font, 7)
        canvas.drawString(15 * mm, 7.5 * mm, f"Patient {patient_id} | Report {report_id}")
        canvas.drawRightString(page_w - 15 * mm, 7.5 * mm, f"Page {report_doc.page}")
        canvas.restoreState()

    story = []
    story.append(Paragraph("Lung Cancer Risk Assessment Report", styles["RptTitle"]))
    story.append(Paragraph("Multimodal analysis of structured clinical data and baseline CT imaging", styles["RptSub"]))

    meta = Table([
        [P("Report ID", "SmallBold"), P(report_id, "SmallRpt"), P("Generated", "SmallBold"), P(generated_at, "SmallRpt")],
        [P("Prepared for", "SmallBold"), P(patient_name, "SmallRpt"), P("System", "SmallBold"), P("Lung Cancer AI Analysis Dashboard", "SmallRpt")],
    ], colWidths=[24 * mm, 60 * mm, 24 * mm, 62 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale),
        ("BOX", (0, 0), (-1, -1), 0.5, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta)
    story.append(Spacer(1, 4 * mm))

    summary = data.get("patient_summary") or {}
    clinical_values = data.get("clinical_values") or {}
    age = summary.get("age", clinical_values.get("age"))
    gender = summary.get("gender") or ("Male" if clinical_values.get("gender") == 0 else ("Female" if clinical_values.get("gender") == 1 else "-"))
    smoker = summary.get("current_smoker") or ("Yes" if clinical_values.get("cigsmok") == 1 else ("No" if clinical_values.get("cigsmok") == 0 else "-"))
    pack_years = clinical_values.get("pack_years", "-")

    story.append(Paragraph("1. Patient information", styles["Section"]))
    patient_table = Table([
        [P("Patient ID", "SmallBold"), P(patient_id), P("Patient name", "SmallBold"), P(patient_name)],
        [P("Age", "SmallBold"), P(value_text(age)), P("Gender", "SmallBold"), P(gender)],
        [P("Current smoker", "SmallBold"), P(smoker), P("Pack-years", "SmallBold"), P(value_text(pack_years))],
    ], colWidths=[29 * mm, 55 * mm, 29 * mm, 57 * mm])
    patient_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, line), ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(patient_table)

    final_prob = data.get("final_prob")
    risk_label = data.get("risk_label") or "-"
    risk_tier = int(data.get("risk_tier", 0) or 0)
    tier_colors = [green, colors.HexColor("#B19B20"), amber, red]
    risk_color = tier_colors[risk_tier] if 0 <= risk_tier < len(tier_colors) else blue
    provenance = data.get("provenance") or {}
    fusion_threshold = provenance.get("fusion_threshold", threshold)
    confidence = data.get("confidence_label") or "-"
    confidence_note = data.get("confidence_note") or ""
    recommendation = data.get("recommendation_text") or "Clinical correlation and appropriate follow-up are recommended."

    story.append(Paragraph("2. Final multimodal assessment", styles["Section"]))
    score_card = Table([[
        Paragraph(pct(final_prob, 1), styles["ScoreBig"]),
        Paragraph(f"MODEL CLASSIFICATION<br/>{risk_label}", styles["WhiteBold"]),
        Paragraph(f"Decision threshold<br/>{num(fusion_threshold, 4)}", styles["WhiteBold"]),
    ]], colWidths=[47 * mm, 76 * mm, 47 * mm], rowHeights=[25 * mm])
    score_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(score_card)
    story.append(Spacer(1, 2.5 * mm))

    assess_table = Table([
        [P("Confidence", "SmallBold"), P(confidence), P("Model threshold", "SmallBold"), P(num(fusion_threshold, 4))],
        [P("Confidence note", "SmallBold"), P(confidence_note), P("Research recommendation", "SmallBold"), P(recommendation)],
    ], colWidths=[27 * mm, 58 * mm, 31 * mm, 54 * mm])
    assess_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale), ("BOX", (0, 0), (-1, -1), 0.5, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(assess_table)

    story.append(Paragraph("3. Model-by-model results", styles["Section"]))
    branch_rows = [[P("Model", "SmallBold"), P("Risk probability", "SmallBold"), P("Fusion weight", "SmallBold"), P("Weighted contribution", "SmallBold")]]
    branch_defs = [
        ("Clinical data model", data.get("clinical_prob"), w_clin),
        ("2D CT model", data.get("old_cnn_prob"), w_2d),
        ("3D MIL model", data.get("mil_prob"), w_mil),
        ("Whole-lung 3D model", data.get("wholelung_prob"), w_whole),
    ]
    whole_available = data.get("wholelung_prob") is not None
    effective_weights = [w_clin, w_2d, w_mil, w_whole]
    if not whole_available:
        s = sum(effective_weights[:3])
        effective_weights = [w / s for w in effective_weights[:3]] + [0.0]
    for idx, (label, prob, nominal_w) in enumerate(branch_defs):
        eff_w = effective_weights[idx]
        contribution = float(prob) * eff_w if prob is not None else None
        weight_label = f"{eff_w * 100:.0f}%" if prob is not None else "Unavailable"
        branch_rows.append([P(label), P(pct(prob, 1)), P(weight_label), P(pct(contribution, 1) if contribution is not None else "-")])
    branch_table = Table(branch_rows, colWidths=[63 * mm, 34 * mm, 31 * mm, 42 * mm], repeatRows=1)
    branch_table.setStyle(table_style())
    story.append(branch_table)

    agreement = data.get("model_agreement") or {}
    spread = agreement.get("spread")
    spread_text = pct(spread, 1) if spread is not None else "-"
    if spread is None:
        agreement_note = "Model agreement could not be summarized."
    elif spread < 0.10:
        agreement_note = "Branch probabilities are relatively close to one another."
    elif spread < 0.25:
        agreement_note = "The branches show moderate variation in their patient-level probabilities."
    else:
        agreement_note = "The branches show a wide probability spread; review the individual model outputs carefully."
    agreement_table = Table([
        [P("Branches available", "SmallBold"), P(agreement.get("n_branches", "-")), P("Probability range", "SmallBold"), P(f"{pct(agreement.get('range_min'), 1)} to {pct(agreement.get('range_max'), 1)}")],
        [P("Branch spread", "SmallBold"), P(spread_text), P("Interpretation", "SmallBold"), P(agreement_note)],
    ], colWidths=[31 * mm, 36 * mm, 31 * mm, 72 * mm])
    agreement_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale), ("BOX", (0, 0), (-1, -1), 0.5, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 2 * mm))
    story.append(agreement_table)

    story.append(Paragraph("4. CT study information", styles["Section"]))
    ct = data.get("ct_study_info") or {}
    spacing = ct.get("pixel_spacing_mm")
    spacing_text = " x ".join(str(x) for x in spacing) + " mm" if isinstance(spacing, list) else "-"
    thickness = ct.get("slice_thickness_mm")
    thickness_text = f"{thickness} mm" if thickness not in (None, "") else "-"
    ct_rows = [
        [P("Files received", "SmallBold"), P(ct.get("files_received")), P("Selected series slices", "SmallBold"), P(ct.get("selected_series_slices"))],
        [P("Modality", "SmallBold"), P(ct.get("modality")), P("Series description", "SmallBold"), P(ct.get("series_description"))],
        [P("Slice thickness", "SmallBold"), P(thickness_text), P("Pixel spacing", "SmallBold"), P(spacing_text)],
        [P("Study date", "SmallBold"), P(ct.get("study_date")), P("3D candidate patches", "SmallBold"), P(data.get("n_3d_patches"))],
    ]
    ct_table = Table(ct_rows, colWidths=[31 * mm, 42 * mm, 38 * mm, 59 * mm])
    ct_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, line), ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(ct_table)

    thumbs = data.get("slice_thumbnails_b64") or []
    chosen = []
    if thumbs:
        # Up to six evenly spaced views so the CT section is useful rather than
        # a single decorative thumbnail.
        if len(thumbs) <= 6:
            indices = list(range(len(thumbs)))
        else:
            indices = sorted(set(round(i * (len(thumbs) - 1) / 5) for i in range(6)))
        for ix in indices:
            img = decode_report_image(thumbs[ix], 50 * mm, 49 * mm)
            if img:
                chosen.append((img, ix + 1, len(thumbs)))
    elif data.get("preview_image_b64"):
        img = decode_report_image(data.get("preview_image_b64"), 80 * mm, 70 * mm)
        if img:
            chosen.append((img, 1, 1))

    if chosen:
        story.append(PageBreak())
        story.append(Paragraph("5. Representative CT views", styles["Section"]))
        story.append(Paragraph(
            "Representative views from the uploaded CT study are shown for report traceability. "
            "These images are visual references only; the dashboard risk score is generated by the complete model pipeline.",
            styles["BodyRpt"],
        ))
        story.append(Spacer(1, 3 * mm))
        for row_start in range(0, len(chosen), 3):
            row_items = chosen[row_start:row_start + 3]
            img_row, cap_row = [], []
            for img, ix, total in row_items:
                img_row.append(img)
                cap_row.append(P(f"CT view {ix}/{total}", "SmallRpt"))
            while len(img_row) < 3:
                img_row.append("")
                cap_row.append("")
            image_table = Table([img_row, cap_row], colWidths=[56 * mm, 56 * mm, 56 * mm])
            image_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(image_table)
            story.append(Spacer(1, 2 * mm))

    story.append(PageBreak())
    story.append(Paragraph("6. Submitted clinical data", styles["Section"]))
    story.append(Paragraph(
        "The table below records the structured values submitted to the clinical branch for this analysis. It is included for traceability and reproducibility of the patient-level result.",
        styles["BodyRpt"],
    ))
    story.append(Spacer(1, 2 * mm))

    preferred = ["age", "gender", "cigsmok", "pack_years"]
    keys = list(clinical_values.keys())
    ordered_keys = [k for k in preferred if k in clinical_values] + sorted(k for k in keys if k not in preferred)
    clinical_rows = [[P("Variable", "SmallBold"), P("Submitted value", "SmallBold"), P("Meaning", "SmallBold")]]
    for key in ordered_keys:
        try:
            desc, expected = describe_column(key)
            meaning = f"{desc}. Expected: {expected}"
        except Exception:
            meaning = key.replace("_", " ").capitalize()
        clinical_rows.append([
            Paragraph(str(key), styles["Appendix"]),
            Paragraph(value_text(clinical_values.get(key)), styles["Appendix"]),
            Paragraph(str(meaning), styles["Appendix"]),
        ])
    if len(clinical_rows) == 1:
        clinical_rows.append([P("-", "SmallRpt"), P("No clinical input values were supplied.", "SmallRpt"), P("-", "SmallRpt")])
    clinical_table = LongTable(clinical_rows, colWidths=[51 * mm, 31 * mm, 88 * mm], repeatRows=1)
    clinical_table.setStyle(table_style(header_bg=blue))
    story.append(clinical_table)

    story.append(PageBreak())
    story.append(Paragraph("7. Analysis provenance", styles["Section"]))
    prov_rows = [
        [P("Analysis timestamp", "SmallBold"), P(provenance.get("analysis_timestamp") or generated_at)],
        [P("Pipeline", "SmallBold"), P(provenance.get("pipeline"))],
        [P("Model configuration", "SmallBold"), P(provenance.get("model_configuration"))],
        [P("Fusion weights", "SmallBold"), P(provenance.get("fusion_weights"))],
        [P("Fusion threshold", "SmallBold"), P(num(provenance.get("fusion_threshold", threshold), 4))],
        [P("Dashboard physician", "SmallBold"), P(f"{cfg.get('doctor_name', '-')} - {cfg.get('doctor_title', '')}")],
    ]
    prov_table = Table(prov_rows, colWidths=[48 * mm, 122 * mm])
    prov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale), ("BOX", (0, 0), (-1, -1), 0.5, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(prov_table)
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=0.7, color=line, spaceBefore=2 * mm, spaceAfter=3 * mm))
    story.append(Paragraph(
        "Research-use statement: This report summarizes the output of a research decision-support system. The reported probabilities and model classification are not a standalone diagnosis and must not be used as a substitute for radiologic interpretation, clinical examination, or appropriate diagnostic follow-up.",
        styles["SmallRpt"],
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    buf.seek(0)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in patient_id) or "patient"
    return buf, f"patient_report_{safe_id}.pdf"


# -----------------------------------------------------------------------------
# Dependency-light fallback PDF generator
# -----------------------------------------------------------------------------
def _build_patient_report_pillow(data, cfg, describe_column, weights, threshold, fallback_reason=None):
    """Generate the same patient report as a multipage PDF using Pillow only.

    This is intentionally kept as a fallback for machines where ReportLab is not
    installed (or where a ReportLab/font-specific error occurs).  The dashboard
    already depends on Pillow for CT previews, so report export remains usable
    without an extra package installation.
    """
    import base64
    import io
    import os
    import textwrap
    from datetime import datetime
    from PIL import Image as PILImage, ImageDraw, ImageFont

    W, H = 1240, 1754  # A4-ish at 150 dpi
    M = 72
    NAVY = (13, 23, 56)
    PINK = (230, 67, 122)
    BLUE = (62, 120, 215)
    GREEN = (45, 159, 111)
    AMBER = (214, 151, 47)
    RED = (201, 68, 97)
    PALE = (245, 247, 252)
    LINE = (221, 226, 238)
    TEXT = (32, 40, 58)
    MUTED = (102, 112, 133)
    WHITE = (255, 255, 255)

    def _font_candidates(bold=False):
        linux = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        mac = '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf'
        mac2 = '/Library/Fonts/Arial Bold.ttf' if bold else '/Library/Fonts/Arial.ttf'
        return [linux, mac, mac2]

    def font(size, bold=False):
        for fp in _font_candidates(bold):
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    F_TITLE = font(34, True)
    F_H1 = font(23, True)
    F_H2 = font(18, True)
    F_BODY = font(14, False)
    F_BODY_B = font(14, True)
    F_SMALL = font(12, False)
    F_SMALL_B = font(12, True)
    F_TINY = font(10, False)
    F_SCORE = font(48, True)

    patient_id = str(data.get('patient_id') or 'patient')
    patient_name = str(data.get('patient_name') or '-')
    generated_at = datetime.now().strftime('%d %b %Y %H:%M')
    report_id = f"LC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    pages = []

    def new_page(title=None):
        im = PILImage.new('RGB', (W, H), WHITE)
        d = ImageDraw.Draw(im)
        d.line((M, H-55, W-M, H-55), fill=LINE, width=2)
        d.text((M, H-42), f'Patient {patient_id} | Report {report_id}', fill=MUTED, font=F_TINY)
        d.text((W-M-120, H-42), f'Page {len(pages)+1}', fill=MUTED, font=F_TINY)
        if title:
            d.text((M, M), title, fill=NAVY, font=F_H1)
            y = M + 45
        else:
            y = M
        pages.append(im)
        return im, d, y

    def wrap(draw, text, fnt, max_width):
        text = '' if text is None else str(text)
        words = text.split()
        if not words:
            return ['-']
        lines, cur = [], ''
        for w in words:
            test = w if not cur else cur + ' ' + w
            if draw.textlength(test, font=fnt) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def draw_wrapped(draw, xy, text, fnt=F_BODY, fill=TEXT, max_width=1000, line_gap=5):
        x, y = xy
        lines = wrap(draw, text, fnt, max_width)
        try:
            bbox = fnt.getbbox('Ag')
            lh = bbox[3]-bbox[1] + line_gap
        except Exception:
            lh = 20 + line_gap
        for line in lines:
            draw.text((x, y), line, fill=fill, font=fnt)
            y += lh
        return y

    def section(draw, y, text):
        draw.rounded_rectangle((M, y, W-M, y+42), radius=10, fill=NAVY)
        draw.text((M+16, y+10), text, fill=WHITE, font=F_H2)
        return y + 56

    def box(draw, x1, y1, x2, y2, fill=PALE, outline=LINE, radius=10):
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill, outline=outline, width=2)

    def pct(prob, digits=1):
        try:
            return 'Unavailable' if prob is None else f'{float(prob)*100:.{digits}f}%'
        except Exception:
            return 'Unavailable'

    def num(v, digits=4):
        try:
            return f'{float(v):.{digits}f}'
        except Exception:
            return '-'

    def value_text(v):
        if v is None or v == '':
            return '-'
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f'{v:.4g}'
        return str(v)

    def draw_kv_grid(draw, y, items, cols=2, row_h=58):
        gap = 12
        total_w = W - 2*M
        cell_w = (total_w - gap*(cols-1))/cols
        rows = (len(items)+cols-1)//cols
        for i, (k, v) in enumerate(items):
            r, c = divmod(i, cols)
            x = M + c*(cell_w+gap)
            yy = y + r*(row_h+gap)
            box(draw, x, yy, x+cell_w, yy+row_h, fill=PALE)
            draw.text((x+12, yy+9), str(k), fill=MUTED, font=F_SMALL_B)
            draw_wrapped(draw, (x+12, yy+29), value_text(v), F_BODY_B, TEXT, cell_w-24, 2)
        return y + rows*(row_h+gap)

    # PAGE 1 -----------------------------------------------------------------
    im, d, y = new_page()
    d.text((M, y), 'Lung Cancer Risk Assessment Report', fill=NAVY, font=F_TITLE)
    y += 52
    d.text((M, y), 'Multimodal analysis of structured clinical data and baseline CT imaging', fill=MUTED, font=F_BODY)
    y += 38
    box(d, M, y, W-M, y+78, fill=PALE)
    d.text((M+15, y+12), f'Report ID: {report_id}', fill=TEXT, font=F_SMALL_B)
    d.text((M+590, y+12), f'Generated: {generated_at}', fill=TEXT, font=F_SMALL_B)
    d.text((M+15, y+43), f'Prepared for: {patient_name}', fill=TEXT, font=F_SMALL)
    d.text((M+590, y+43), 'System: Lung Cancer AI Analysis Dashboard', fill=TEXT, font=F_SMALL)
    y += 100

    summary = data.get('patient_summary') or {}
    clinical_values = data.get('clinical_values') or {}
    age = summary.get('age', clinical_values.get('age'))
    gv = clinical_values.get('gender')
    gender = summary.get('gender') or ('Male' if gv == 0 else ('Female' if gv == 1 else '-'))
    sv = clinical_values.get('cigsmok')
    smoker = summary.get('current_smoker') or ('Yes' if sv == 1 else ('No' if sv == 0 else '-'))
    pack_years = clinical_values.get('pack_years', '-')

    y = section(d, y, '1. Patient information')
    y = draw_kv_grid(d, y, [
        ('Patient ID', patient_id), ('Patient name', patient_name), ('Age', age), ('Gender', gender),
        ('Current smoker', smoker), ('Pack-years', pack_years)
    ], cols=2)
    y += 8

    final_prob = data.get('final_prob')
    risk_label = data.get('risk_label') or '-'
    risk_tier = int(data.get('risk_tier', 0) or 0)
    risk_colors = [GREEN, (177,155,32), AMBER, RED]
    risk_color = risk_colors[risk_tier] if 0 <= risk_tier < len(risk_colors) else BLUE
    provenance = data.get('provenance') or {}
    fusion_threshold = provenance.get('fusion_threshold', threshold)
    confidence = data.get('confidence_label') or '-'
    confidence_note = data.get('confidence_note') or ''
    recommendation = data.get('recommendation_text') or 'Clinical correlation and appropriate follow-up are recommended.'

    y = section(d, y, '2. Final multimodal assessment')
    box(d, M, y, W-M, y+150, fill=risk_color, outline=risk_color, radius=14)
    d.text((M+24, y+23), pct(final_prob, 1), fill=WHITE, font=F_SCORE)
    d.text((M+360, y+26), 'MODEL CLASSIFICATION', fill=WHITE, font=F_SMALL_B)
    d.text((M+360, y+58), str(risk_label), fill=WHITE, font=F_H1)
    d.text((M+800, y+28), 'Decision threshold', fill=WHITE, font=F_SMALL_B)
    d.text((M+800, y+58), num(fusion_threshold, 4), fill=WHITE, font=F_H1)
    d.text((M+24, y+112), f'Confidence: {confidence}', fill=WHITE, font=F_BODY_B)
    y += 172
    box(d, M, y, W-M, y+145, fill=PALE)
    d.text((M+14, y+12), 'Interpretation', fill=NAVY, font=F_BODY_B)
    yy = draw_wrapped(d, (M+14, y+38), confidence_note, F_BODY, TEXT, W-2*M-28, 4)
    d.text((M+14, max(yy+6, y+86)), 'Research recommendation', fill=NAVY, font=F_BODY_B)
    draw_wrapped(d, (M+14, max(yy+30, y+110)), recommendation, F_BODY, TEXT, W-2*M-28, 4)

    # PAGE 2 -----------------------------------------------------------------
    im, d, y = new_page('3. Model-by-model results')
    w_clin, w_2d, w_mil, w_whole = weights
    branch_defs = [
        ('Clinical data model', data.get('clinical_prob'), w_clin),
        ('2D CT model', data.get('old_cnn_prob'), w_2d),
        ('3D MIL model', data.get('mil_prob'), w_mil),
        ('Whole-lung 3D model', data.get('wholelung_prob'), w_whole),
    ]
    effective = [w_clin, w_2d, w_mil, w_whole]
    if data.get('wholelung_prob') is None:
        s = sum(effective[:3])
        effective = [w/s for w in effective[:3]] + [0]
    headers = ['Model', 'Risk probability', 'Fusion weight', 'Weighted contribution']
    xcols = [M, 470, 720, 930, W-M]
    row_h = 62
    d.rectangle((M, y, W-M, y+45), fill=NAVY)
    for i, h in enumerate(headers):
        d.text((xcols[i]+10, y+13), h, fill=WHITE, font=F_SMALL_B)
    y += 45
    for i, (label, prob, _) in enumerate(branch_defs):
        eff_w = effective[i]
        bg = WHITE if i%2==0 else PALE
        d.rectangle((M, y, W-M, y+row_h), fill=bg, outline=LINE)
        vals = [label, pct(prob), f'{eff_w*100:.0f}%' if prob is not None else 'Unavailable', pct(float(prob)*eff_w) if prob is not None else '-']
        for c, val in enumerate(vals):
            draw_wrapped(d, (xcols[c]+10, y+18), val, F_BODY, TEXT, xcols[c+1]-xcols[c]-20, 2)
        y += row_h

    agreement = data.get('model_agreement') or {}
    y += 25
    y = section(d, y, '4. Model agreement')
    spread = agreement.get('spread')
    if spread is None:
        agreement_note = 'Model agreement could not be summarized.'
    elif spread < 0.10:
        agreement_note = 'Branch probabilities are relatively close to one another.'
    elif spread < 0.25:
        agreement_note = 'The branches show moderate variation in their patient-level probabilities.'
    else:
        agreement_note = 'The branches show a wide probability spread; review individual model outputs carefully.'
    y = draw_kv_grid(d, y, [
        ('Branches available', agreement.get('n_branches', '-')),
        ('Probability range', f"{pct(agreement.get('range_min'))} to {pct(agreement.get('range_max'))}"),
        ('Branch spread', pct(spread)),
        ('Interpretation', agreement_note),
    ], cols=2, row_h=84)

    y += 10
    y = section(d, y, '5. CT study information')
    ct = data.get('ct_study_info') or {}
    spacing = ct.get('pixel_spacing_mm')
    spacing_text = ' x '.join(str(x) for x in spacing) + ' mm' if isinstance(spacing, list) else '-'
    thickness = ct.get('slice_thickness_mm')
    y = draw_kv_grid(d, y, [
        ('Files received', ct.get('files_received')),
        ('Selected series slices', ct.get('selected_series_slices')),
        ('Modality', ct.get('modality')),
        ('Series description', ct.get('series_description')),
        ('Slice thickness', f'{thickness} mm' if thickness not in (None,'') else '-'),
        ('Pixel spacing', spacing_text),
        ('Study date', ct.get('study_date')),
        ('3D candidate patches', data.get('n_3d_patches')),
    ], cols=2, row_h=70)

    # PAGE 3 representative CT views -----------------------------------------
    thumbs = data.get('slice_thumbnails_b64') or []
    if thumbs or data.get('preview_image_b64'):
        im, d, y = new_page('6. Representative CT views')
        y = draw_wrapped(d, (M, y),
            'Representative views from the uploaded CT study are shown for traceability. These images are visual references only; the risk score is generated by the complete model pipeline.',
            F_BODY, TEXT, W-2*M, 5) + 24
        sources = []
        if thumbs:
            idxs = list(range(len(thumbs))) if len(thumbs)<=6 else sorted(set(round(i*(len(thumbs)-1)/5) for i in range(6)))
            sources = [(thumbs[ix], ix+1, len(thumbs)) for ix in idxs]
        else:
            sources = [(data.get('preview_image_b64'), 1, 1)]
        cell_w, cell_h = 335, 315
        gap_x, gap_y = 30, 55
        for j, (b64, ix, total) in enumerate(sources[:6]):
            r, c = divmod(j, 3)
            x = M + c*(cell_w+gap_x)
            yy = y + r*(cell_h+gap_y)
            box(d, x, yy, x+cell_w, yy+cell_h, fill=(15,18,28), outline=LINE)
            try:
                raw = base64.b64decode(b64)
                ci = PILImage.open(io.BytesIO(raw)).convert('RGB')
                ci.thumbnail((cell_w-18, cell_h-45))
                px = x+(cell_w-ci.width)//2
                py = yy+10+(cell_h-45-ci.height)//2
                im.paste(ci, (px, py))
                d.text((x+10, yy+cell_h-28), f'CT view {ix}/{total}', fill=WHITE, font=F_SMALL)
            except Exception:
                d.text((x+20, yy+30), 'CT view unavailable', fill=MUTED, font=F_BODY)

    # Clinical data pages -----------------------------------------------------
    preferred = ['age', 'gender', 'cigsmok', 'pack_years']
    keys = list(clinical_values.keys())
    ordered = [k for k in preferred if k in clinical_values] + sorted(k for k in keys if k not in preferred)
    rows_per_page = 20
    if not ordered:
        ordered = ['-']
    for start in range(0, len(ordered), rows_per_page):
        im, d, y = new_page('7. Submitted clinical data' + (f' (continued)' if start else ''))
        if start == 0:
            y = draw_wrapped(d, (M, y), 'Structured values submitted to the clinical branch are listed for patient-level traceability and reproducibility.', F_BODY, TEXT, W-2*M, 4) + 22
        headers = ['Variable', 'Submitted value', 'Meaning']
        xcols = [M, 380, 620, W-M]
        d.rectangle((M, y, W-M, y+43), fill=BLUE)
        for i,h in enumerate(headers):
            d.text((xcols[i]+10, y+12), h, fill=WHITE, font=F_SMALL_B)
        y += 43
        for ridx, key in enumerate(ordered[start:start+rows_per_page]):
            val = clinical_values.get(key) if key != '-' else 'No clinical input values were supplied.'
            try:
                desc, expected = describe_column(key)
                meaning = f'{desc}. Expected: {expected}'
            except Exception:
                meaning = key.replace('_',' ').capitalize() if key!='-' else '-'
            rh = 62
            bg = WHITE if ridx%2==0 else PALE
            d.rectangle((M, y, W-M, y+rh), fill=bg, outline=LINE)
            vals = [key, value_text(val), meaning]
            for c,v in enumerate(vals):
                draw_wrapped(d, (xcols[c]+8, y+10), v, F_SMALL, TEXT, xcols[c+1]-xcols[c]-16, 2)
            y += rh

    # Provenance final page ---------------------------------------------------
    im, d, y = new_page('8. Analysis provenance')
    y = draw_kv_grid(d, y, [
        ('Analysis timestamp', provenance.get('analysis_timestamp') or generated_at),
        ('Pipeline', provenance.get('pipeline')),
        ('Model configuration', provenance.get('model_configuration')),
        ('Fusion weights', provenance.get('fusion_weights')),
        ('Fusion threshold', num(provenance.get('fusion_threshold', threshold), 4)),
        ('Dashboard physician', f"{cfg.get('doctor_name','-')} - {cfg.get('doctor_title','')}"),
    ], cols=1, row_h=70)
    y += 25
    y = section(d, y, 'Research-use statement')
    draw_wrapped(d, (M, y),
        'This report summarizes the output of a research decision-support system. The reported probabilities and model classification are not a standalone diagnosis and must not be used as a substitute for radiologic interpretation, clinical examination, or appropriate diagnostic follow-up.',
        F_BODY, TEXT, W-2*M, 6)
    if fallback_reason:
        d.text((M, H-90), 'PDF engine: compatibility mode (Pillow)', fill=MUTED, font=F_TINY)

    buf = io.BytesIO()
    pages[0].save(buf, format='PDF', save_all=True, append_images=pages[1:], resolution=150.0)
    buf.seek(0)
    safe_id = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in patient_id) or 'patient'
    return buf, f'patient_report_{safe_id}.pdf'


def build_patient_report(data, cfg, describe_column, weights, threshold):
    """Generate the patient PDF with automatic compatibility fallback.

    Preferred engine: ReportLab (vector text/tables).
    Fallback engine: Pillow multipage PDF, requiring no additional dashboard
    package beyond the Pillow dependency already used for CT previews.
    """
    try:
        return _build_patient_report_reportlab(data, cfg, describe_column, weights, threshold)
    except Exception as exc:
        # Do not make PDF export fail just because ReportLab is absent or a
        # machine-specific font/encoding issue occurred.
        print(f'[patient-report] ReportLab path failed; using Pillow fallback: {type(exc).__name__}: {exc}')
        return _build_patient_report_pillow(data, cfg, describe_column, weights, threshold, fallback_reason=str(exc))
