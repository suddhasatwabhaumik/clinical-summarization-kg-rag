# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compiles Clinical KG-RAG research manuscript into an academic LaTeX document.
Reads generated prose sections, compiles empirical BigQuery tables, embeds charts,
and outputs a complete, compile-ready .tex manuscript.
"""

import os
import re
from typing import Dict, Any
import pandas as pd
from src.generate_charts import generate_research_charts


def escape_underscores_outside_math(text: str) -> str:
    r"""
    Escapes underscores in regular prose, leaving math blocks untouched.
    Supports inline math ($...$) and environment math (\begin{equation}...\end{equation}).
    Guarantees that already-escaped underscores (\_) are not double-escaped.
    """
    # First protect equation blocks
    eq_pattern = r'(\\begin\{equation\}.*?\\end\{equation\})'
    eq_blocks = []
    def eq_repl(m):
        idx = len(eq_blocks)
        eq_blocks.append(m.group(0))
        return f"QQQEQBLOCK{idx}QQQ"
    text = re.sub(eq_pattern, eq_repl, text, flags=re.DOTALL)

    # Now handle inline math ($...$)
    parts = text.split("$")
    for i in range(len(parts)):
        if i % 2 == 0:  # Outside math mode block
            # First normalize any double backslash underscores to standard single underscore
            parts[i] = parts[i].replace("\\\\_", "_")
            parts[i] = parts[i].replace("\\_", "_")
            # Now escape all underscores cleanly once
            parts[i] = parts[i].replace("_", "\\_")
    text = "$".join(parts)

    # Restore equation blocks
    for idx, block in enumerate(eq_blocks):
        text = text.replace(f"QQQEQBLOCK{idx}QQQ", block)
    return text


def read_clean_section(name: str, sections_dir: str = "output/latex/sections") -> str:
    """Reads prose section text, strips Markdown headings, and formats bold text and math to LaTeX."""
    path = os.path.join(sections_dir, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Convert bold markdown to LaTeX
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    # Format inequalities for math mode
    text = re.sub(r'([A-Za-z0-9_]+)\s*≤\s*([0-9.]+)', r'$\1 \\le \2$', text)
    text = text.replace("≤", r"$\le$")
    text = text.replace("≠", r"$\neq$")

    # Safe escape literal %
    text = re.sub(r'(?<!\\)%', r'\\%', text)
    # Safe escape literal #
    text = re.sub(r'(?<!\\)#', r'\\#', text)
    # Safe escape underscores outside math
    text = escape_underscores_outside_math(text)

    return text


def build_system_architecture_diagram() -> str:
    """Builds a formal TikZ vector diagram for the dual-path KG-RAG architecture."""
    return r"""
\begin{figure}[H]
  \centering
  \resizebox{0.86\textwidth}{!}{
  \begin{tikzpicture}[
    box/.style={rectangle, draw=black, thick, rounded corners, minimum height=1.0cm, minimum width=6.8cm, align=center, fill=blue!6, font=\small},
    db/.style={cylinder, draw=black, thick, shape border rotate=90, aspect=0.22, minimum height=1.7cm, minimum width=3.3cm, align=center, fill=green!12, font=\small},
    llm/.style={rectangle, draw=black, thick, rounded corners, minimum height=1.0cm, minimum width=6.8cm, align=center, fill=orange!15, font=\small},
    arrow/.style={-{Stealth[scale=1.1]}, thick}
  ]
    % Central and branch nodes with explicit coordinates
    \node[box] (ehr) at (0, 0) {Unstructured Clinical Notes\\(MIMIC-IV Inpatient Discharge Summaries)};
    \node[box] (pre) at (0, -1.5) {Data Preprocessing \& Clinical NER\\(\texttt{scispacy} / Regex De-Identification Masking)};

    \node[db] (chroma) at (-3.4, -3.8) {Vector Store\\(ChromaDB)\\Dense 512-Token Chunks\\(PubMedBERT $d=768$)};
    \node[db] (graph) at (3.4, -3.8) {Serverless Graph Store\\(Google BigQuery)\\UMLS Concept Triples\\(2-Hop Communities)};

    \node[box, fill=purple!10] (hybrid) at (0, -6.3) {Dual-Path Hybrid Context Fusion Engine\\(Semantic Text Chunks + Multi-Hop Graph Subgraphs)};
    \node[box, fill=yellow!15] (prompt) at (0, -7.7) {Graph-of-Thought (GoT) Constrained Prompt Synthesis\\(Deterministic Ontological Guardrails \& Rules)};
    \node[llm] (model) at (0, -9.1) {Generative Synthesis Engine\\(Google Vertex AI Gemini 3 Pro)};
    \node[box, fill=green!20] (summary) at (0, -10.5) {Structured Clinical Handover Summary\\(Zero Allergy Omissions, Verified Diagnostic Timeline)};

    % Top branching flows
    \draw[arrow] (ehr) -- (pre);
    \draw[arrow] (pre.south) -- ++(0,-0.35) -| node[above, pos=0.75, font=\footnotesize] {Tokenize \& Embed} (chroma.north);
    \draw[arrow] (pre.south) -- ++(0,-0.35) -| node[above, pos=0.75, font=\footnotesize] {CUI Link \& Ingest} (graph.north);

    % Intermediate fusion flows with clean rectangular orthogonal paths
    \draw[arrow] (chroma.south) -- ++(0,-0.45) -| node[above, pos=0.25, font=\footnotesize] {Top-$k$ Chunks} ([xshift=-2.0cm]hybrid.north);
    \draw[arrow] (graph.south) -- ++(0,-0.45) -| node[above, pos=0.25, font=\footnotesize] {2-Hop Subgraphs} ([xshift=2.0cm]hybrid.north);

    % Bottom linear pipeline
    \draw[arrow] (hybrid) -- (prompt);
    \draw[arrow] (prompt) -- (model);
    \draw[arrow] (model) -- (summary);
  \end{tikzpicture}
  }
  \caption{Dual-path neuro-symbolic KG-RAG architecture integrating dense semantic vector retrieval (ChromaDB / PubMedBERT) with multi-hop ontological graph reasoning (BigQuery / UMLS) and Gemini 3 Pro generative synthesis.}
  \label{fig:architecture}
\end{figure}
"""


def generate_latex_manuscript(
    output_path: str = "output/latex/clinical_kgrag_manuscript.tex"
) -> None:
    """Compiles sections, figures, tables, and references into the final LaTeX manuscript."""
    print("Generating experimental charts...")
    generate_research_charts()

    print("Reading prose sections...")
    abstract = read_clean_section("abstract")
    intro = read_clean_section("introduction")
    theory = read_clean_section("theory")
    methodology = read_clean_section("methodology")
    graph_rag_implementation = read_clean_section("graph_rag_implementation")
    metrics = read_clean_section("metrics")
    results = read_clean_section("results")
    discussion = read_clean_section("discussion")
    conclusion = read_clean_section("conclusion")

    latex_content = f"""% --- ACADEMIC MANUSCRIPT PREAMBLE ---
\\documentclass[11pt, a4paper]{{article}}
\\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.2cm, right=2.2cm]{{geometry}}
\\usepackage[english]{{babel}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{float}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{graphicx}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning, arrows.meta, shapes.geometric, calc}}
\\usepackage{{hyperref}}
\\usepackage{{enumitem}}
\\usepackage{{microtype}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black
}}

\\title{{\\textbf{{Mitigating Hallucinations in Clinical Summarization:\\\\Knowledge Graphs and Retrieval-Augmented Generation}}}}

\\author{{
  \\textbf{{Suddhasatwa Bhaumik}} \\\\
  \\texttt{{suddhasatwa@google.com}} \\\\
  \\small GitHub Repository: \\href{{https://github.com/suddhasatwabhaumik/clinical-summarization-kg-rag}}{{github.com/suddhasatwabhaumik/clinical-summarization-kg-rag}}
}}

\\date{{September 2026}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{intro}

\\section{{Theoretical Background and Related Work}}
{theory}

\\section{{Experimental Design and Methodology}}
{methodology}

\\section{{Knowledge Graph Construction and Hybrid Graph-RAG Engineering}}
{graph_rag_implementation}

{build_system_architecture_diagram()}

\\section{{Evaluation Framework and Metrics}}
{metrics}

\\section{{Empirical Results and Analysis}}
{results}

\\section{{Discussion and Practical Implications}}
{discussion}

\\section{{Conclusion}}
{conclusion}

\\begin{{thebibliography}}{{99}}

\\bibitem{{agrawal2022}}
Agrawal, M., Hegselmann, S., Lang, H., Kim, Y., \\& Sontag, D. (2022).
Large language models are few-shot clinical information extractors.
\\textit{{Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)}}, 1998--2022.

\\bibitem{{alsentzer2019}}
Alsentzer, E., Murphy, J. R., Boag, W., Weng, W. H., Jin, D., Naumann, T., \\& McDermott, M. (2019).
Publicly available clinical BERT embeddings.
\\textit{{arXiv preprint arXiv:1904.03323}}.

\\bibitem{{asgari2024}}
Asgari, E., Monta\\~na-Brown, N., Dubois, M., Khalil, S., Balloch, J., \\& Pimenta, D. (2024).
A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation.
\\textit{{medRxiv}}. \\url{{https://doi.org/10.1101/2024.01.15.24301321}}

\\bibitem{{gu2021}}
Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Chiang, D., ... \\& Poon, H. (2021).
Domain-specific language model pretraining for biomedical natural language processing.
\\textit{{ACM Transactions on Computing for Healthcare}}, 3(1), 1--23.

\\bibitem{{jiang2024}}
Jiang, P., Xiao, C., Jiang, M., Bhatia, P., Kass-Hout, T., Sun, J., \\& Han, J. (2024).
Reasoning-enhanced healthcare predictions with knowledge graph community retrieval.
\\textit{{arXiv preprint arXiv:2403.03912}}.

\\bibitem{{johnson2023}}
Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., ... \\& Mark, R. G. (2023).
MIMIC-IV, a freely accessible electronic health record dataset.
\\textit{{Scientific Data}}, 10(1), 1.

\\bibitem{{lewis2020}}
Lewis, P., Perez, E., Piktus, A., Petroni, F., Lewis, V., K\\\"uttler, M., \\& Kiela, D. (2020).
Retrieval-augmented generation for knowledge-intensive NLP tasks.
\\textit{{Advances in Neural Information Processing Systems (NeurIPS)}}, 33, 9459--9474.

\\bibitem{{shickel2018}}
Shickel, B., Tighe, P. J., Bihorac, A., \\& Rashidi, P. (2018).
Deep EHR: A survey on recent advances in deep learning techniques for electronic health record data.
\\textit{{IEEE Journal of Biomedical and Health Informatics}}, 22(5), 1589--1604.

\\bibitem{{singhal2023}}
Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., ... \\& Natarajan, V. (2023).
Large language models encode clinical knowledge.
\\textit{{Nature}}, 620(7972), 172--180.

\\bibitem{{yasunaga2021}}
Yasunaga, M., Bosselut, A., Liang, P., \\& Leskovec, J. (2021).
QA-GNN: Reasoning with language models and knowledge graphs for question answering.
\\textit{{Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)}}, 535--546.

\\bibitem{{yasunaga2022}}
Yasunaga, M., Leskovec, J., \\& Liang, P. (2022).
LinkBERT: Pretraining language models with document links.
\\textit{{Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)}}, 8003--8016.

\\bibitem{{zhang2023}}
Zhang, R., Tsai, H. R., \\& Mei, Q. (2023).
Clinical text summarization with large language models: A systematic review.
\\textit{{Journal of Biomedical Informatics}}, 145, 104462.

\\end{{thebibliography}}

\\end{{document}}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"LaTeX manuscript successfully compiled to {output_path}")


if __name__ == "__main__":
    generate_latex_manuscript()
