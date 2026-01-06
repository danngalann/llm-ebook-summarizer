"""PDF content extraction module with hierarchical TOC support."""

import fitz  # PyMuPDF

from utils import sanitize_filename


class TOCNode:
    """Represents a node in the table of contents tree."""
    
    def __init__(self, level: int, title: str, page: int, idx: int):
        self.level = level
        self.title = title
        self.page = page
        self.idx = idx  # Sequential index for ordering
        self.end_page = None
        self.children = []
        self.text = None
        self.summary = None
        self.parent = None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0
    
    def get_filename(self) -> str:
        """Generate filename for this node."""
        sanitized = sanitize_filename(self.title, max_length=60) if self.title else "untitled"
        return f"{self.idx:03d}_{sanitized}"
    
    def __repr__(self):
        return f"TOCNode(level={self.level}, title='{self.title}', page={self.page}, children={len(self.children)})"


def build_toc_tree(toc: list) -> list:
    """
    Builds a hierarchical tree structure from a flat TOC list.
    
    Args:
        toc (list): List of [level, title, page] entries from PDF TOC.
    
    Returns:
        list: List of root-level TOCNode objects with children populated.
    """
    if not toc:
        return []
    
    # Create all nodes
    nodes = []
    for idx, (level, title, page) in enumerate(toc, start=1):
        node = TOCNode(level, title, page, idx)
        nodes.append(node)
    
    # Set end_page for each node
    for i, node in enumerate(nodes):
        # Find next node at same or higher level
        for j in range(i + 1, len(nodes)):
            if nodes[j].level <= node.level:
                node.end_page = nodes[j].page
                break
        # If no next node found, end_page will be set later to doc length
    
    # Build hierarchy
    roots = []
    stack = []  # Stack of (level, node) tuples
    
    for node in nodes:
        # Pop stack until we find the parent level
        while stack and stack[-1][0] >= node.level:
            stack.pop()
        
        if stack:
            # Add as child to the node on top of stack
            parent = stack[-1][1]
            parent.children.append(node)
            node.parent = parent
        else:
            # This is a root node
            roots.append(node)
        
        # Push current node onto stack
        stack.append((node.level, node))
    
    return roots


def extract_text_for_node(doc, node: TOCNode, doc_length: int) -> str:
    """
    Extracts text content for a specific TOC node.
    
    Args:
        doc: PyMuPDF document object.
        node (TOCNode): The node to extract text for.
        doc_length (int): Total number of pages in document.
    
    Returns:
        str: Extracted text content.
    """
    start_page = node.page - 1  # Convert to 0-indexed
    end_page = (node.end_page - 1) if node.end_page else doc_length
    
    text = ""
    for page_num in range(start_page, end_page):
        if page_num < doc_length:
            page = doc[page_num]
            text += page.get_text()
    
    return text.strip()


def extract_text_for_page_range(doc, start_page: int, end_page: int, doc_length: int) -> str:
    """
    Extracts text content for a specific page range.
    
    Args:
        doc: PyMuPDF document object.
        start_page (int): Start page (1-indexed, as in TOC).
        end_page (int): End page (1-indexed, exclusive).
        doc_length (int): Total number of pages in document.
    
    Returns:
        str: Extracted text content.
    """
    start_idx = start_page - 1  # Convert to 0-indexed
    end_idx = end_page - 1
    
    text = ""
    for page_num in range(start_idx, end_idx):
        if 0 <= page_num < doc_length:
            page = doc[page_num]
            text += page.get_text()
    
    return text.strip()


def get_ancestor_intro_texts(doc, node: TOCNode, doc_length: int) -> str:
    """
    Collects introductory text from ancestor nodes, only if this leaf
    is the "first child" descendant of that ancestor.
    
    For each ancestor, extracts text between the ancestor's start page
    and its first child's start page. This captures any introductory
    content before the nested sections begin.
    
    The intro is only included if this leaf is reachable from the ancestor
    by always following the first child. This prevents the same intro
    from being repeated for sibling sections.
    
    Args:
        doc: PyMuPDF document object.
        node (TOCNode): The leaf node to get ancestor intros for.
        doc_length (int): Total number of pages in document.
    
    Returns:
        str: Combined introductory text from ancestors where this is the first descendant.
    """
    intro_texts = []
    current = node
    
    # Walk up the tree, checking if we're the first child at each level
    while current.parent is not None:
        parent = current.parent
        
        # Only include this parent's intro if current is the first child
        if parent.children and parent.children[0] == current:
            first_child = parent.children[0]
            if parent.page < first_child.page:
                intro = extract_text_for_page_range(
                    doc, parent.page, first_child.page, doc_length
                )
                if intro:
                    # Prepend with section title for context
                    intro_texts.insert(0, f"[Context from: {parent.title}]\n{intro}")
        else:
            # Not the first child, stop collecting intros from higher ancestors
            break
        
        current = parent
    
    return "\n\n".join(intro_texts)


def collect_leaf_nodes(roots: list) -> list:
    """
    Collects all leaf nodes from the tree in depth-first order.
    
    Args:
        roots (list): List of root TOCNode objects.
    
    Returns:
        list: List of leaf TOCNode objects.
    """
    leaves = []
    
    def traverse(node):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                traverse(child)
    
    for root in roots:
        traverse(root)
    
    return leaves


def extract_pdf_content(pdf_path: str, min_word_count: int = 200) -> list:
    """
    Extracts chapters from a PDF file using its table of contents.
    Only processes leaf nodes (innermost sections), but includes
    introductory context from ancestor nodes.
    
    Args:
        pdf_path (str): Path to the PDF file.
        min_word_count (int): Minimum word count for a chapter to be included.
    
    Returns:
        list: List of dicts with 'node' and 'text' for LLM processing.
    
    Raises:
        ValueError: If the PDF has no table of contents/outline.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Returns list of [level, title, page]
    
    if not toc:
        doc.close()
        raise ValueError(
            f"PDF file '{pdf_path}' has no table of contents. "
            "Only PDFs with chapters/outline are supported."
        )
    
    # Build hierarchical tree
    tree_roots = build_toc_tree(toc)
    
    # Get all leaf nodes
    leaf_nodes = collect_leaf_nodes(tree_roots)
    
    # Extract text for each leaf node with ancestor context
    doc_length = len(doc)
    leaf_data = []
    
    for node in leaf_nodes:
        # Get introductory text from ancestors
        ancestor_intro = get_ancestor_intro_texts(doc, node, doc_length)
        
        # Get the leaf's own text
        leaf_text = extract_text_for_node(doc, node, doc_length)
        
        # Combine: ancestor intro + leaf text
        if ancestor_intro:
            combined_text = f"{ancestor_intro}\n\n---\n\n{leaf_text}"
        else:
            combined_text = leaf_text
        
        words = combined_text.split()
        
        if len(words) >= min_word_count:
            node.text = ' '.join(words)
            leaf_data.append({
                'node': node,
                'text': node.text
            })
    
    doc.close()
    return leaf_data
