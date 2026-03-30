# Document Loading & Chunking

## Overview
Explored different document sources and chunking strategies to prepare text for RAG systems.

## What's Built

### Document Loaders
- **`web_loader.py`**: Loads and cleans text from web pages
  - Removes script/style elements
  - Extracts clean text content
  - Handles HTML parsing with BeautifulSoup

- **`pdf_loader.py`**: Extracts text from PDF files
  - Uses pypdf for text extraction
  - Handles multi-page documents

### Chunking Strategies
Tested 3 approaches to split documents:

| Strategy | Method | Best For |
|----------|--------|----------|
| **Fixed Size** | Split by character count (500 chars, 50 overlap) | Simple, predictable chunks |
| **Fixed Words** | Split by word count (100 words, 20 overlap) | Preserving word boundaries |
| **Semantic** | Split on paragraphs/sentences (200-500 chars) | Preserving meaning and context |

### Evaluation Method
Instead of metrics, used actual LLM generation with Mistral to evaluate chunking quality:
- Retrieved top-3 relevant chunks per query
- Generated answers using retrieved context
- Compared answer quality across strategies

## Results

### Web Document Test
Loaded content from: *https://en.wikipedia.org/wiki/Ice_dance#Falls_and_interruptions*

**Query:**
Give four reasons that can cause interruption of an ice dance?

## Example Output
**STRATEGY**: Fixed Size (500 chars, 50 overlap)

ANSWER:
 1. Interruptions can occur due to the skaters themselves falling or experiencing a fall (one point for one partner and two points if both partners fall).

2. Interruptions can also happen due to an interruption lasting between 10 seconds and less than 20 seconds, for which ice dancers can lose one point.

3. An interruption can occur if it lasts between 20 and 30 seconds, resulting in a loss of two points for the ice dance team.

4. Adverse conditions such as lighting issues, ice conditions, or items thrown onto the ice can cause an interruption in the performance. In such cases, skaters may stop their routine until the problem is resolved. Additionally, if an adverse condition lasts over ten minutes, a second warm-up takes place before continuing the competition.

---
**STRATEGY**: Fixed Words (100 words, 20 overlap)

ANSWER:
 Four reasons that can cause interruption in an ice dance, as per the provided context, are:

1. One or both partners fall during their performance.
2. An adverse condition unrelated to the skaters and/or their equipment such as lighting or ice problems.
3. An interruption ordered by the Referee for reasons not specified in the given context.
4. Interruptions lasting over 20 seconds, whether caused by falls, adverse conditions, or orders from the referee, which result in the program being resumed from the point of interruption or the start of the element, if applicable.

---
**STRATEGY**: Semantic (200-500 chars)

ANSWER:
 1. Technical problems, such as issues with music, equipment, or lighting, which can occur over 20 seconds after the start of their program.
2. Interruptions during performance due to unforeseen circumstances like falls, injuries, or wardrobe malfunctions that last more than ten seconds but not over twenty seconds.
3. Unresolved conflict between partners, which can cause consistent disruptions in their partnership and potentially lead to the early break-up of a team.
4. Exceeding the allowed time limit for an interruption (more than thirty seconds without resuming, or more than forty seconds within the first three minutes).
