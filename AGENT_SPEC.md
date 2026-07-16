# Repository Specification for AI Agents

This document defines the constraints, guidelines, and formatting rules for AI agents reading, editing, or managing this repository. 

---

## 1. Core Rule: Content Integrity
- **Do not modify the actual content, prose, or code examples** of the notes in this repository unless explicitly instructed by the user. 
- You may perform index updates, link corrections, and structure fixes as requested, but the text of the study notes must remain completely untouched.

---

## 2. Markdown Link Compatibility Specification
To ensure that all links work perfectly and render correctly in both **Obsidian** and **GitHub**, follow these exact link formatting rules:

### A. Relative Links and File Extensions
- **Always use relative paths** starting from the current file's directory. Do not use absolute paths.
- **Always include the `.md` file extension** for file links (e.g., `[Topic](01%20-%20Topic.md)`). This is required by GitHub to render the target file correctly in the web interface.

### B. Spaces and URL Encoding
- **Spaces MUST be encoded as `%20`**: Standard Markdown renderers (including GitHub's) break link recognition at raw space characters. Therefore, spaces in paths must always be replaced with `%20`.
- **Do NOT use angled brackets `<...>` inside parentheses**: Avoid formats like `[Label](<path with spaces/file.md>)`. While allowed in standard CommonMark, Obsidian's visual link resolution, backlinks, and editor indexing do not support them correctly.
- **Keep all other characters unencoded**: For readability, do not quote/encode other special characters such as commas `,`, ampersands `&`, and parentheses `(`, `)` (e.g. keep `(C1).md` and `&` as they are). Standard Markdown parsers in both GitHub and Obsidian support these raw characters in paths as long as spaces are encoded.

*Example:*
- **❌ Incorrect (WikiLink)**: `[[C1 - Supervised Machine Learning/00 Index (C1)]]` (Breaks on GitHub)
- **❌ Incorrect (Angled brackets)**: `[Index](<C1 - Supervised Machine Learning/00 Index (C1).md>)` (Breaks in Obsidian)
- **❌ Incorrect (Aggressive URL encoding)**: `[Index](C1%20-%20Supervised%20Machine%20Learning/00%20Index%20%28C1%29.md)` (Hard to read)
- **✔️ Correct (Simple URL-encoded space)**: `[Index](C1%20-%20Supervised%20Machine%20Learning/00%20Index%20(C1).md)` (Works everywhere, clean)

### C. Header/Anchor Links
- Internal section links within the same file must be prefixed with `#` and follow the standard GitHub Flavored Markdown (GFM) header slugification rules:
  1. Convert all characters to lowercase.
  2. Strip punctuation (such as `:`, `.`, `,`, etc.).
  3. Replace spaces and consecutive spaces with hyphens `-`.
- *Example:* A section titled `## Coursera: Advanced Learning Algorithms` resolves to `[Text](#coursera-advanced-learning-algorithms)`.
