# notes2audio

`notes2audio` is a Python-based pipeline that converts PDF and DOCX study notes into high-quality, "listenable" audio files (MP3). Unlike simple Text-to-Speech (TTS) tools, it uses Large Language Models (LLMs) to rewrite messy, bulleted notes into natural, fluid spoken-word scripts before synthesizing them.

Check out the [examples/](examples/) directory to see the transformation from raw notes to refined audio.

## 🚀 Features

- **LLM-Powered Rewriting**: Automatically converts bullet points, abbreviations, and messy formatting into coherent sentences optimized for listening.
- **Context-Aware Processing**: Breaks long documents into chunks while maintaining context by sharing summaries of preceding and succeeding parts with the LLM.
- **Parallel Processing**: Speeds up the conversion by processing multiple chunks simultaneously with concurrency limits.
- **High-Quality TTS**: Uses Microsoft's `edge-tts` for natural-sounding neural voices.
- **Cost Tracking**: Logs estimated API costs for rewriting (supports OpenRouter/Gemini).
- **Flexible Configuration**: Easily customize prompts, voices, speech rates, and models via `config.yaml`.
- **Debug Mode**: Saves intermediate steps (extracted text, cleaned chunks, prompts) for inspection.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tomsouri/notes2audio.git
   cd notes2audio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key**:
   The script uses [OpenRouter](https://openrouter.ai/) for LLM processing. Export your API key as an environment variable:
   ```bash
   export OPENROUTER_API_KEY='your_api_key_here'
   ```

## ⚙️ Configuration

Modify `config.yaml` to adjust the behavior:

- **api**: Set the LLM model (default: `openai/gpt-4o-mini`), timeout, and temperature.
- **prompts**: Customize how the AI summarizes and rewrites your notes.
- **processing**: Tune chunk sizes (`max_api_chunk_chars`, `max_tts_chunk_chars`) and concurrency (`max_concurrent_tasks`).
- **defaults**: Set your preferred language, voice (e.g., `cs-CZ-AntoninNeural`), and speech rate.

## 📖 Usage

### Basic Usage
Convert a single PDF or DOCX file:
```bash
python run.py notes.pdf
python run.py notes.docx
```

### Multiple Files or Directories
```bash
# Multiple files
python run.py chapter1.pdf notes.docx

# Whole directory (finds all PDFs and DOCX files)
python run.py ./study_materials/
```

### Advanced Options
```bash
# Use a specific voice and slower speed
python run.py --voice cs-CZ-AntoninNeural --rate "-15%" notes.pdf

# Specify output directory
python run.py --output-dir ./audio_notes notes.pdf

# Only run TTS on already cleaned text (skips LLM step)
python run.py --only_synthesize notes.pdf
```

## 📂 Project Structure

- `run.py`: Main entry point and pipeline logic.
- `config.yaml`: Configuration for API, prompts, and defaults.
- `requirements.txt`: Python package dependencies.
- `examples/`: Example files showing input, intermediate text, and output audio.
- `[file]_chunks/`: (Auto-generated) Contains raw chunks, LLM prompts, and cleaned text for debugging.
- `[file].extracted.txt`: (Auto-generated) The raw text extracted from the PDF or DOCX.
- `[file].txt`: (Auto-generated) The final rewritten script used for audio.

## 📝 How it Works

1. **Extraction**: Uses `PyMuPDF` for PDFs and `python-docx` for DOCX files to pull raw text.
2. **Summarization**: Generates brief summaries for every chunk in parallel to provide context.
3. **Rewriting**: Sends each chunk to an LLM along with the "neighboring" summaries. The AI rewrites the notes into a narrative script.
4. **Synthesis**: Uses `edge-tts` to convert the rewritten scripts into MP3 parts.
5. **Assembly**: Saves individual MP3 parts and the final rewritten script.

## ⚖️ License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See the [LICENSE](LICENSE) file for the full text.
