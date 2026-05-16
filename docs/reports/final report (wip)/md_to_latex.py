"""Convert the markdown report to LaTeX."""
import re

with open("PREPARE_Final_Report.md", "r") as f:
    md = f.read()

lines = md.split("\n")

tex = []
tex.append(r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{float}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{caption}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{\textbf{Prediction of Prices in Agriculture (PREPARE)} \\ \vspace{0.5cm} \large Final Project Report}
\author{
    Varun SG \quad Kunal Ranjan \quad Aryan Gosain \quad Rishit Anand \\[0.3cm]
    \textit{Plaksha University} \\[0.3cm]
    \textbf{Faculty Guide:} Prof.\ Mayank Ratan Bhardwaj
}
\date{9 May 2026}

\begin{document}

\maketitle

\begin{center}
\begin{tabular}{ll}
\toprule
\textbf{Name} & \textbf{Role / Work Distribution} \\
\midrule
Varun SG & Data collection, scraping, EDA \\
Kunal Ranjan & Imputation pipeline, feature engineering \\
Aryan Gosain & Model development, graph architectures \\
Rishit Anand & App development, evaluation, integration \\
\bottomrule
\end{tabular}
\end{center}

\vspace{1cm}

\begin{abstract}
Agricultural price volatility is one of the most pressing challenges for India's small and marginal farmers, who lack reliable tools to anticipate market shifts. PREPARE (Prediction of Prices in Agriculture) addresses this problem by building a multi-day commodity price forecasting system for four key crops (onion, potato, tomato, and wheat) across hundreds of mandis (agricultural markets) nationwide.

We scraped three years of daily mandi-level price and arrival data (January 2023 to December 2025) from the Agmarknet portal and developed an imputation pipeline to handle the 65--75\% structural missingness inherent in government-reported agricultural datasets. We then conducted a broad model comparison spanning naive baselines, gradient-boosted tabular models, cross-crop feature augmentation, density- and volume-aware variants, and deep spatiotemporal graph neural networks.

Our best-performing architecture, a Graph Attention Network with GRU temporal encoding (GAT-GRU), achieved 15-day $R^2$ scores of 0.81 (onion), 0.94 (potato), 0.39 (tomato), and 0.70 (wheat), substantially exceeding the previous-day baseline (which achieves an $R^2$ of only 0.94 for same-day prediction on onion) and all non-graph alternatives. The final models have been integrated into a bilingual mobile application built with FastAPI, enabling farmers and regulators to access real-time, location-specific price forecasts for horizons of 1 to 15 days.

\noindent\textbf{Keywords:} agricultural price forecasting, graph neural networks, GAT-GRU, mandi networks, spatiotemporal modeling
\end{abstract}

\newpage
\tableofcontents
\newpage
""")

# Now convert the body sections
# Skip everything up to "## 1. Introduction"
i = 0
while i < len(lines):
    if lines[i].strip().startswith("## 1. Introduction"):
        break
    i += 1

in_itemize = False
in_enumerate = False
enum_counter = 0
figure_counter = 0

def escape_tex(s):
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("$", r"\$")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s

def process_inline(s):
    """Process inline markdown: bold, italic, code, links."""
    # Bold **text**
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    # Italic *text*  (but not inside \textbf)
    # Code `text`
    s = re.sub(r'`([^`]+)`', r'\\texttt{\1}', s)
    # Links [text](url)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', s)
    return s

def tex_escape_light(s):
    """Light escape: only chars that break LaTeX but preserve our commands."""
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    # Don't escape _ in \texttt or \href
    # Handle _ outside of commands
    result = []
    in_cmd = 0
    ci = 0
    while ci < len(s):
        if s[ci] == '\\' and ci+1 < len(s) and s[ci+1:ci+5] in ['text', 'href', 'begi', 'end{', 'item', 'topr', 'midr', 'bott', 'tabu', 'mult', 'hlin', 'newp']:
            result.append(s[ci])
        elif s[ci] == '{':
            in_cmd += 1
            result.append(s[ci])
        elif s[ci] == '}':
            in_cmd -= 1
            result.append(s[ci])
        elif s[ci] == '_' and in_cmd == 0:
            result.append(r'\_')
        elif s[ci] == '#' and in_cmd == 0:
            result.append(r'\#')
        elif s[ci] == '~' and in_cmd == 0:
            result.append(r'\textasciitilde{}')
        else:
            result.append(s[ci])
        ci += 1
    return ''.join(result)

# Process line by line from section 1 onwards
in_table = False
table_lines = []

def close_list():
    global in_itemize, in_enumerate
    r = []
    if in_itemize:
        r.append(r"\end{itemize}")
        in_itemize = False
    if in_enumerate:
        r.append(r"\end{enumerate}")
        in_enumerate = False
    return r

def flush_table(table_lines):
    """Convert collected markdown table lines to LaTeX tabular."""
    if len(table_lines) < 2:
        return []
    header = table_lines[0]
    data_rows = table_lines[2:]  # skip separator line
    
    cols = [c.strip() for c in header.split("|")[1:-1]]
    ncols = len(cols)
    
    result = []
    col_spec = "l" * ncols
    result.append(r"\begin{table}[H]")
    result.append(r"\centering")
    result.append(r"\begin{tabular}{" + col_spec + "}")
    result.append(r"\toprule")
    
    header_cells = []
    for c in cols:
        c = c.replace("**", "")
        c = process_inline(c)
        c = tex_escape_light(c)
        header_cells.append(r"\textbf{" + c + "}")
    result.append(" & ".join(header_cells) + r" \\")
    result.append(r"\midrule")
    
    for row in data_rows:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        processed = []
        for c in cells:
            c = c.replace("**", "")
            c = process_inline(c)
            c = tex_escape_light(c)
            processed.append(c)
        result.append(" & ".join(processed) + r" \\")
    
    result.append(r"\bottomrule")
    result.append(r"\end{tabular}")
    result.append(r"\end{table}")
    return result

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Skip markdown horizontal rules
    if stripped == "---":
        i += 1
        continue
    
    # Handle figures
    if stripped.startswith("!["):
        match = re.match(r'!\[(.+?)\]\((.+?)\)', stripped)
        if match:
            caption = match.group(1)
            path = match.group(2)
            tex.extend(close_list())
            tex.append(r"\begin{figure}[H]")
            tex.append(r"\centering")
            tex.append(r"\includegraphics[width=0.85\textwidth]{" + path + "}")
            tex.append(r"\caption{" + tex_escape_light(process_inline(caption)) + "}")
            tex.append(r"\end{figure}")
        i += 1
        continue
    
    # Handle headings
    if stripped.startswith("## "):
        tex.extend(close_list())
        title = stripped[3:]
        title = re.sub(r'^\d+\.\s*', '', title)
        tex.append(r"\section{" + process_inline(title) + "}")
        i += 1
        continue
    
    if stripped.startswith("### "):
        tex.extend(close_list())
        title = stripped[4:]
        title = re.sub(r'^\d+\.\d+\s*', '', title)
        tex.append(r"\subsection{" + process_inline(title) + "}")
        i += 1
        continue
    
    # Handle tables
    if "|" in stripped and not stripped.startswith("-") and not stripped.startswith("!["):
        if not in_table:
            tex.extend(close_list())
            in_table = True
            table_lines = []
        table_lines.append(stripped)
        i += 1
        continue
    elif in_table:
        if stripped.startswith("|"):
            table_lines.append(stripped)
            i += 1
            continue
        else:
            tex.extend(flush_table(table_lines))
            table_lines = []
            in_table = False
            # Don't increment i, process this line normally
            continue
    
    # Handle ordered lists
    m = re.match(r'^(\d+)\.\s+(.+)', stripped)
    if m:
        if not in_enumerate:
            tex.extend(close_list())
            in_enumerate = True
            tex.append(r"\begin{enumerate}")
        item_text = m.group(2)
        item_text = process_inline(item_text)
        item_text = tex_escape_light(item_text)
        tex.append(r"\item " + item_text)
        i += 1
        continue
    
    # Handle unordered lists
    if stripped.startswith("- "):
        if not in_itemize:
            tex.extend(close_list())
            in_itemize = True
            tex.append(r"\begin{itemize}")
        item_text = stripped[2:]
        item_text = process_inline(item_text)
        item_text = tex_escape_light(item_text)
        tex.append(r"\item " + item_text)
        i += 1
        continue
    
    # Close lists if we hit a non-list line
    if (in_itemize or in_enumerate) and stripped and not stripped.startswith("- ") and not re.match(r'^\d+\.', stripped):
        tex.extend(close_list())
    
    # Empty line
    if not stripped:
        i += 1
        continue
    
    # Bold-only lines (like "**Evaluation Protocol:**")
    if stripped.startswith("**") and stripped.endswith("**"):
        inner = stripped[2:-2]
        tex.append(r"\noindent\textbf{" + tex_escape_light(inner) + "}")
        tex.append("")
        i += 1
        continue
    
    # Regular paragraph
    pline = process_inline(stripped)
    pline = tex_escape_light(pline)
    tex.append(pline)
    tex.append("")
    i += 1

# Close any open lists
tex.extend(close_list())

# Flush any remaining table
if in_table and table_lines:
    tex.extend(flush_table(table_lines))

# Add references section manually
tex.append(r"""
\section{References}

\begin{enumerate}
\item Bhardwaj, M. R. (2023). \textit{Novel Algorithms for Improving Agricultural Planning and Operations using Artificial Intelligence and Game Theory.} Doctoral dissertation, Indian Institute of Science, Bangalore.

\item Yu, B., Yin, H., and Zhu, Z. (2018). Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. In \textit{Proceedings of IJCAI 2018}.

\item Wu, Z., Pan, S., Long, G., Jiang, J., and Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. In \textit{Proceedings of IJCAI 2019}.

\item Li, Y., Yu, R., Shahabi, C., and Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. In \textit{Proceedings of ICLR 2018}.

\item Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., and Bengio, Y. (2018). Graph Attention Networks. In \textit{Proceedings of ICLR 2018}.

\item Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In \textit{Proceedings of KDD 2016}.

\item Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In \textit{Advances in Neural Information Processing Systems (NeurIPS) 2017}.

\item Meena, M., et al. (2025). Self-Adaptive Graph Mixture of Models. \textit{arXiv preprint arXiv:2511.13062}.

\item Agmarknet Portal. Government of India, Ministry of Agriculture and Farmers' Welfare. \url{https://agmarknet.gov.in/}
\end{enumerate}

\end{document}
""")

# Write output
output = "\n".join(tex)

# Post-process: fix double-escaped issues
output = output.replace(r"\\_", r"\_")
output = output.replace(r"\\%", r"\%")
output = output.replace(r"\\&", r"\&")
output = output.replace(r"\\#", r"\#")
# Fix R-squared to R^2
output = output.replace("R-squared", "$R^2$")
# Fix >= 
output = output.replace(">=", "$\\geq$")

with open("r1.tex", "w") as f:
    f.write(output)

print("Done: r1.tex written")
