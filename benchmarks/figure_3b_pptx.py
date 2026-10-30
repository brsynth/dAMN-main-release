#!/usr/bin/env python
"""
Figure 3b as a PowerPoint slide: acetate predicted by dFBA, MLP and dAMN on
the Millard data (models trained on glucose and biomass only), in the style of
Fig3.pptx (10 x 5.625 in, Arial 11 pt, same model colours). Curves and data
are a native (editable) XY chart; the +-1 sd bands of MLP and dAMN are
semi-transparent shapes drawn behind it, at positions computed from the chart's
fixed plot area and axis ranges, and grouped with it. The method text is in
the slide notes.

Reads results/millard/figure_3b_curves.csv (figure_3b_data.py).

Usage:
    python figure_3b_pptx.py <output .pptx>
"""

import sys

import pandas as pd
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

FONT = 'Arial'
TEXT = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
GRID = RGBColor(0xE3, 0xE2, 0xDE)
COLORS = {  # Fig3.pptx series colours
    'dFBA': RGBColor(0xF5, 0xA9, 0x8A),
    'MLP': RGBColor(0x86, 0xB6, 0xEF),
    'dAMN': RGBColor(0x7F, 0xD3, 0xB2),
}
NOTES = '\n'.join([
    'Figure 3b. Prediction of acetate overflow in E. coli. Data: Millard et al. (2021), '
    'strain MG1655 in M9 with 12.7 mM glucose and 0.96 mM acetate at t = 0, 13 time points '
    'over 4.6 h. The three models were trained or fitted on the glucose and biomass time '
    'series only; acetate measurements were never used, except its initial concentration '
    '(0.96 mM) given as initial condition. The curves are the predicted extracellular acetate '
    'concentrations; dots are the measured values. R² = 1 - SSE/SST over the 13 acetate '
    'measurements.',
    '',
    'MLP (no mechanism). Feed-forward network t -> Dense(500, ReLU) -> Dropout(0.2) -> '
    'Dense(3) giving glucose, acetate and biomass at time t, written as '
    'C(t) = C(0) + s * [f(t) - f(0)] so that every output starts at its measured initial '
    'value (s: per-variable scale). Trained with Adam (learning rate 1e-3, 3000 full-batch '
    'epochs) on the mean-squared error of glucose and biomass, each scaled by its standard '
    'deviation; the acetate output is excluded from the loss and receives no training signal. '
    'Line: mean of 3 random seeds; shaded: +- 1 s.d. over the seeds. Predicted acetate '
    'stays at its initial value (R² < 0).',
    '',
    'dAMN. Published model of the dAMN paper (SI Fig. S9b; release v1.0.0 of this repository, '
    'training mode "concentration-ACE"), not retrained here: the curve was digitised from the '
    'published plot (ensemble mean of 3 models). dAMN integrates the concentrations '
    'C(t) = (glucose, acetate, biomass) step by step from C(0). A neural network (one hidden '
    'layer of 500 ReLU units, dropout 0.2) maps C(t) to the 3682 fluxes v of the E. coli '
    'genome-scale model iML1515; the concentration change is the product of v with the '
    'exchange (transport) matrix, scaled by a lag-phase factor given by a second network '
    '(50 hidden units). Loss: concentration error on glucose and biomass (acetate masked), '
    'with weight decaying over time (rate 0.25), + stoichiometric violation ||S v||² '
    '(weight 0.01) + negative-flux penalty (weight 1) + a penalty on acetate increases larger '
    'than 0.3 mM per time step (UB_out = 0.3). Adam (learning rate 1e-3, exponential decay), '
    'batch size 5, up to 1000 epochs with early stopping (patience 100); one pre-training run '
    'followed by 3 successive training runs, each kept as an ensemble member. Line: '
    'ensemble mean; shaded: +- 1 s.d. over the 3 members (both read from the published plot). '
    'Acetate R² = 0.94.',
    '',
    'dFBA. Standard dynamic FBA, static optimisation approach (Varma & Palsson 1994; '
    'Mahadevan et al. 2002), with iML1515 (default ATP maintenance) in COBRApy. At each 0.05 h '
    'step, glucose and acetate uptake are bounded by Michaelis-Menten kinetics, '
    'v <= Vmax C / (Km + C) (Km,glc = 0.01 mM, Km,ac = 0.2 mM, Vmax,ac = 10 mmol/gDW/h), '
    'and by the amount left in the medium; O2 uptake is bounded by O2max; growth is maximised '
    'and fluxes are made unique by parsimonious FBA; biomass and concentrations are then '
    'integrated exactly over the step. Acetate secretion is unconstrained. Vmax,glc and O2max '
    'were set by grid search (Vmax,glc 4-10.5, O2max 6-25 mmol/gDW/h) minimising the glucose '
    'and biomass error only, acetate not used: Vmax,glc = 8 and O2max = 15 mmol/gDW/h (the '
    'latter equals the measured maximum of Varma & Palsson 1994). Acetate R² = 0.73. On a '
    'finer grid, other parameter sets fit glucose and biomass equally well (R² >= 0.98 for '
    'both) but give acetate R² from -5.7 to 0.73: glucose and biomass alone do not fix how '
    'dFBA splits surplus carbon between CO2 and acetate.',
])
PANELS = [  # (variable, title, y axis title)
    ('glc__D_e', 'Glucose (training fit)', 'Glucose (mM)'),
    ('BIOMASS', 'Biomass (training fit)', 'Biomass (gDW/L)'),
    ('ac_e', 'Acetate (prediction)', 'Acetate (mM)'),
]


def style_text(font, size=11, bold=False, color=TEXT):
    font.name, font.size, font.bold = FONT, Pt(size), bold
    font.color.rgb = color


def add_text(slide, text, x, y, w, h, size=11, bold=False, color=TEXT,
             align=PP_ALIGN.LEFT, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    style_text(run.font, size, bold, color)
    return box


PLOT = dict(x=0.12, y=0.12, w=0.83, h=0.72)  # inner plot area, fraction of chart frame
X_MAX = 5


def set_manual_plot_area(chart):
    """Pin the inner plot area so data coordinates map to known slide positions."""
    plot_area = chart._chartSpace.find(qn('c:chart')).find(qn('c:plotArea'))
    layout = plot_area.find(qn('c:layout'))
    if layout is None:
        layout = etree.SubElement(plot_area, qn('c:layout'))
        plot_area.insert(0, layout)
    for child in list(layout):
        layout.remove(child)
    ml = etree.SubElement(layout, qn('c:manualLayout'))
    for tag, val in [('layoutTarget', 'inner'), ('xMode', 'edge'), ('yMode', 'edge'),
                     ('x', PLOT['x']), ('y', PLOT['y']), ('w', PLOT['w']), ('h', PLOT['h'])]:
        etree.SubElement(ml, qn(f'c:{tag}')).set('val', str(val))


def add_band(shapes, t, lo, hi, color, frame, y_max, name):
    """Filled +-sd band between lo and hi, in slide coordinates of the plot area."""
    fx, fy, fw, fh = frame
    px = lambda tt: fx + (PLOT['x'] + PLOT['w'] * tt / X_MAX) * fw
    py = lambda v: fy + (PLOT['y'] + PLOT['h'] * (1 - v / y_max)) * fh
    pts = [(px(a), py(b)) for a, b in zip(t, hi)] + \
          [(px(a), py(b)) for a, b in zip(t[::-1], lo[::-1])]
    ff = shapes.build_freeform(Emu(int(pts[0][0])), Emu(int(pts[0][1])), scale=1.0)
    ff.add_line_segments([(Emu(int(a)), Emu(int(b))) for a, b in pts[1:]], close=True)
    band = ff.convert_to_shape()
    band.name = name
    band.fill.solid()
    band.fill.fore_color.rgb = color
    band.fill._xPr.find(qn('a:solidFill')).find(qn('a:srgbClr')).append(
        etree.Element(qn('a:alpha'), val='35000'))
    band.line.fill.background()


def add_panel(shapes, tab, var, title, ytitle, x, y, w, h, y_max, bands=()):
    frame = (Inches(x), Inches(y), Inches(w), Inches(h))
    t = tab['time'].to_numpy()
    for name in bands:  # behind the chart
        mean, sd = tab[f'{name}_{var}'].to_numpy(), tab[f'{name}_{var}_sd'].to_numpy()
        add_band(shapes, t, mean - sd, mean + sd, COLORS[name], frame, y_max,
                 f'{name} sd band')
    data = XyChartData()
    for name in ('dFBA', 'MLP', 'dAMN'):
        s = data.add_series(name)
        for t, v in zip(tab['time'], tab[f'{name}_{var}']):
            s.add_data_point(t, v)
    s = data.add_series('Measured')
    for t, v in zip(tab['time'], tab[f'measured_{var}']):
        s.add_data_point(t, v)
    gf = shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, *frame, data)
    gf.name = f'Figure 3b {title}'
    chart = gf.chart
    chart.has_legend = False
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    style_text(chart.chart_title.text_frame.paragraphs[0].runs[0].font, 12, True)
    for ser in chart.plots[0].series:
        ser.smooth = False
        if ser.name == 'Measured':
            ser.format.line.fill.background()
            ser.marker.style = XL_MARKER_STYLE.CIRCLE
            ser.marker.size = 6
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = TEXT
            ser.marker.format.line.fill.background()
        else:
            ser.format.line.color.rgb = COLORS[ser.name]
            ser.format.line.width = Pt(1.5)
            ser.marker.style = XL_MARKER_STYLE.NONE
    for axis, label in ((chart.category_axis, 'Time (h)'), (chart.value_axis, ytitle)):
        axis.has_title = True
        axis.axis_title.text_frame.text = label
        style_text(axis.axis_title.text_frame.paragraphs[0].runs[0].font, 11)
        style_text(axis.tick_labels.font, 11)
        axis.format.line.color.rgb = GRID
        axis.has_major_gridlines = axis is chart.value_axis
        if axis.has_major_gridlines:
            axis.major_gridlines.format.line.color.rgb = GRID
    chart.category_axis.minimum_scale = 0
    chart.category_axis.maximum_scale = X_MAX
    chart.category_axis.major_unit = 1
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = y_max
    chart.value_axis.major_unit = 1
    set_manual_plot_area(chart)


def add_legend(slide, x, y, labels=None):
    items = [('dFBA', COLORS['dFBA']), ('MLP', COLORS['MLP']), ('dAMN', COLORS['dAMN']),
             ('Measured', TEXT)]
    labels = labels or {}
    for name, color in items:
        if name == 'Measured':
            key = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.02), Inches(y + 0.06),
                                         Inches(0.11), Inches(0.11))
        else:
            key = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.1),
                                         Inches(0.3), Inches(0.04))
        key.name = f'Legend key {name}'
        key.fill.solid()
        key.fill.fore_color.rgb = color
        key.line.fill.background()
        key.shadow.inherit = False  # no theme shadow on legend keys
        off = 0.2 if name == 'Measured' else 0.38
        text = labels.get(name, name)
        width = 0.95 if name == 'Measured' else 0.65 + 0.075 * (len(text) - len(name))
        add_text(slide, text, x + off, y, width, 0.24, name=f'Legend {name}')
        x += off + width


def main():
    out = sys.argv[1]
    tab = pd.read_csv('results/millard/figure_3b_curves.csv')
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(10), Inches(5.625)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    sc = pd.read_csv('results/millard/figure_3b_scores.csv').set_index('model')['R2_ac_e']
    r2 = lambda m: 'R² < 0' if sc[m] < 0 else f'R² = {sc[m]:.2f}'
    add_legend(slide, 1.55, 0.2, {m: f'{m} ({r2(m)})' for m in ('dFBA', 'MLP', 'dAMN')})
    var, title, ytitle = PANELS[2]  # acetate only
    group = slide.shapes.add_group_shape()
    group.name = 'Figure 3b acetate'
    add_panel(group.shapes, tab, var, title, ytitle, 2.0, 0.6, 6.0, 4.2, y_max=6,
              bands=('MLP', 'dAMN'))
    add_text(slide,
             'Models trained on glucose and biomass only; acetate never seen in training. '
             'Shaded: \u00b1 1 s.d. over 3 models (MLP, dAMN).',
             1.0, 4.95, 8.0, 0.3, size=10, color=MUTED, align=PP_ALIGN.CENTER,
             name='Footnote')
    slide.notes_slide.notes_text_frame.text = NOTES
    prs.save(out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
