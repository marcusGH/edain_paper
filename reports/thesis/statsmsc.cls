% statsmsc.cls file created by Dean Bodenham (2021), 
% partially based on Paul Ginzberg's template (2015).
% Using part of icmathsthesis to sort out fonts for chapter/section headings
% Last updated April 2022

% Imperial College's logo from Imperial College's website

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{statsmsc}[2021/09/24 Thesis template for MSc in Statistics]

\LoadClass[11pt,headings=optiontohead,twoside=false]{scrbook}

%%% For formatting text and spacing nicely %%%

% Use the setspace package instead because it only affects text
% \usepackage{setspace} 
% \onehalfspacing % Remove this if report becomes longer than 70 pages

% For not doing indent on new paragraphs
\usepackage{parskip} 

% % Makes text nicer to read
\usepackage[activate={true,nocompatibility}, % > activate protrusion and expansion
final,          % > enable microtype; use "draft" to disable
tracking=true,  % v
kerning=true,   % v
spacing=true,   % > activate these techniques
factor=1100,    % > add 10% to the protrusion amount (default is 1000)
stretch=10,     % v
shrink=10,      % > reduce stretchability/shrinkability (default is 20/20) 
]{microtype}
\microtypecontext{spacing=nonfrench}

%%% End text formatting packages %%%

\RequirePackage{scrtime}
\RequirePackage{setspace}
% From icmathsthesis.cls
\RequirePackage{ifthen}

% Remove blank page between chapters
\KOMAoptions{twoside=false}

% AMS packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}
\usepackage{mathtools}

% for bold symbols, e.g. bold Greek symbols, use the bm package
\usepackage{bm} 
% For enumerations
\RequirePackage{enumerate}
% For margin notes
\RequirePackage{marginnote}
% For \mathghost
\RequirePackage{halloweenmath}

% Graphics, importing figures
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}

% for tables
\RequirePackage{booktabs}

% for positioning Imperial logo
\RequirePackage[absolute]{textpos}

% for bibtex
\RequirePackage[round,authoryear,sort]{natbib}
%\usepackage[authoryear]{natbib}
%\usepackage[numbers]{natbib}

% for clickable table of content entries
\usepackage{color}
\usepackage{xcolor}
\PassOptionsToPackage{hyphens}{url}% to line-break long URLs
\usepackage{hyperref}
\definecolor{linkcolor}{rgb}{0.0,0.0,0.3}
\hypersetup{
    colorlinks=true, %set true if you want colored links
    linktoc=all,     %set to all if you want both sections and subsections linked
    linkcolor=black,  %choose some color if you want links to stand out
    citecolor=linkcolor,
}
\usepackage{cleveref} 
% For handling acronyms
\usepackage{acronym} 

%% Theorem environment and numbering
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\newtheorem{theorem}[definition]{Theorem}
\newtheorem{proposition}[definition]{Proposition}
\newtheorem{lemma}[definition]{Lemma}
\newtheorem{corollary}[definition]{Corollary}
%% making example same as definition style
\newtheorem{example}[definition]{Example}
\newtheorem{remark}[definition]{Remark}

% fixing proof environment
\renewenvironment{proof}[1][\proofname]%
{\noindent\trivlist\item\ignorespaces{\bfseries #1. }}%
    {\hfill $\square$}

%a command for quoting:
\newcommand*{\usequote}[1]{``{#1}''}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Title page
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Title style
\renewcommand{\titlefont}{\normalcolor \normalfont \bfseries \Huge}

%% Title page
%\makeatletter
\newcommand*{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand*{\CID}[1]{\gdef\@CID{#1}}
\newcommand*{\logoimg}{\gdef\@logoimg}
%\newcommand*{\logoimg}[1]{\gdef\@logoimg{#1}}

\renewcommand{\maketitle}{
% set to front matter
\frontmatter


\begin{titlepage}
\ifdefined\@logoimg
\begin{textblock*}{8cm}(1.75cm,1.75cm)
\includegraphics[width=70mm]{figures/imperial_logo.pdf}
\end{textblock*}
\vspace*{1cm}
\else
%\vspace*{0cm}
\fi
\begin{center}
\vspace*{\stretch{0.1}}
%{\large Imperial College London}\\
{\large Department of Mathematics}\par
\vspace*{\stretch{1}} % This inserts vertical space and allows you to specify a relative size for the vertical spaces.
{\titlefont \@title\par} % If your title is long, you may wish to use \huge instead of \Huge.
\vspace*{\stretch{2}}
{\Large \@author \par}
\vspace*{1em}
{\large CID: \@CID \par}
\vspace*{\stretch{0.5}}
{\large Supervised by \@supervisor \par}
\vspace*{\stretch{3}}
{\Large \@date \par}
\vspace*{\stretch{1}}
{\large Submitted in partial fulfilment of the requirements for the
MSc in Statistics of Imperial College London}
\vspace*{\stretch{0.1}}
\end{center}%
\end{titlepage}%
}
\makeatother


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Declaration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcommand*{\declarationdate}[1]{\gdef\@declarationdate{#1}}
\newcommand*{\declarationname}[1]{\gdef\declarationname{#1}}

% Plagiarism delcaration
\newcommand*{\declaration}{%

\newpage
\thispagestyle{plain}
\hphantom{1}
\vfill
\noindent The work contained in this thesis is my own work unless
otherwise stated.
\vfill
{Signed: {\declarationname}  \hfill} 
{\hfill  Date: {\@declarationdate} \hfill}
\vfill
%setting the declaration page to start on page i
\setcounter{page}{1}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Abstract
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{abstract}%
{\chapter*{Abstract}\thispagestyle{plain}}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Acknowledgements
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{acknowledgements}%
{\chapter*{Acknowledgements}\thispagestyle{plain}}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Header style
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage[headsepline,plainheadsepline]{scrlayer-scrpage}
\clearpairofpagestyles
\automark{chapter}
\ohead*{\pagemark}
% "*" means it is included on the chapter page, 
% no "* means it is not included on that page
\ihead{\headmark}
\addtokomafont{pageheadfoot}{\normalfont\bfseries}
\addtokomafont{pagenumber}{\normalfont\bfseries}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Section fonts
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% used koma commands to set the sections to use the same font as title
% had to manually specify font sizes
% if this is not included, different font types are used

\setkomafont{chapter}{
    \usefontofkomafont{title}%
    \huge%
}

\setkomafont{section}{
    \usefontofkomafont{title}%
    \Large%
}

\setkomafont{subsection}{
    \usefontofkomafont{title}%
    \large%
}

\setkomafont{subsubsection}{
    \usefontofkomafont{title}%
    \normalsize%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     References/Bibliography
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% change Bibliography to References
\renewcommand{\bibname}{References}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%      Math commands
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\R}{\mathbb{R}}
\newcommand{\bfX}{\mathbf{X}}
\newcommand{\bfx}{\mathbf{x}}
\newcommand{\bfxvec}{\vec{\mathbf{x}}}
\newcommand{\bfY}{\mathbf{Y}}
\newcommand{\bfy}{\mathbf{y}}
\newcommand{\bfz}{\mathbf{z}}
\newcommand{\bfh}{\mathbf{h}}
\newcommand{\bfZ}{\mathbf{Z}}
\newcommand{\bftheta}{\mathbf{\theta}}

\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
