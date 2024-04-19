import argparse
import asyncio
import concurrent.futures
import datetime
import os
import re
import subprocess
import warnings


from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import edge_tts
from mutagen import mp4
# import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment

warnings.filterwarnings("ignore", module="ebooklib.epub")

_CONCURRENT_THREADS = 20


def chap2text_epub(chap):
    blacklist = [
        "[document]",
        "noscript",
        "header",
        "html",
        "meta",
        "head",
        "input",
        "script",
    ]
    paragraphs = []
    soup = BeautifulSoup(chap, "html.parser")

    # Extract chapter title (assuming it's in an <h1> tag)
    chapter_title = soup.find("h1")
    if chapter_title:
        chapter_title_text = chapter_title.text.strip()
    else:
        chapter_title_text = None

    # Always skip reading links that are just a number (footnotes)
    for a in soup.findAll("a", href=True):
        if not any(char.isalpha() for char in a.text):
            a.extract()

    chapter_paragraphs = soup.find_all("p")
    for p in chapter_paragraphs:
        paragraph_text = "".join(p.strings).strip()
        paragraphs.append(paragraph_text)

    return chapter_title_text, paragraphs


def export(book, sourcefile) -> str:
    book_contents = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapter_title, chapter_paragraphs = chap2text_epub(
                item.get_content())
            book_contents.append(
                {"title": chapter_title, "paragraphs": chapter_paragraphs})
    outfile = sourcefile.replace(".epub", ".txt")
    check_for_file(outfile)
    print(f"Exporting {sourcefile} to {outfile}")
    author = book.get_metadata("DC", "creator")[0][0]
    booktitle = book.get_metadata("DC", "title")[0][0]

    with open(outfile, "w") as file:
        file.write(f"Title: {booktitle}\n")
        file.write(f"Author: {author}\n\n")
        for i, chapter in enumerate(book_contents, start=1):
            if chapter["paragraphs"] == [] or chapter["paragraphs"] == ['']:
                continue
            else:
                if chapter["title"] == None:
                    file.write(f"# Part {i}\n")
                else:
                    file.write(f"# {chapter['title']}\n\n")
                for paragraph in chapter["paragraphs"]:
                    file.write(f"{paragraph}\n\n")

    return outfile


def get_book(sourcefile):
    book_contents = []
    book_title = sourcefile
    book_author = "Unknown"
    chapter_titles = []
    with open(sourcefile, "r", encoding="utf-8") as file:
        current_chapter = {"title": None, "paragraphs": []}
        lines_skipped = 0
        for line in file:
            if lines_skipped < 2 and (line.startswith("Title") or line.startswith("Author")):
                lines_skipped += 1
                if line.startswith('Title: '):
                    book_title = line.replace('Title: ', '').strip()
                elif line.startswith('Author: '):
                    book_author = line.replace('Author: ', '').strip()
                continue
            line = line.strip()
            if line.startswith("#"):
                if current_chapter["paragraphs"]:
                    book_contents.append(current_chapter)
                    current_chapter = {"title": None, "paragraphs": []}
                current_chapter["title"] = line[1:].strip()
                chapter_titles.append(current_chapter["title"])
            elif line:
                current_chapter["paragraphs"].append(line)

        if current_chapter["paragraphs"]:
            book_contents.append(current_chapter)

    return book_contents, book_title, book_author, chapter_titles


def sort_key(s):
    # extract number from the string
    return int(re.findall(r'\d+', s)[0])


def check_for_file(filename):
    if os.path.isfile(filename):
        print(f"The file '{filename}' already exists.")
        overwrite = input("Do you want to overwrite the file? (y/n): ")
        if overwrite.lower() != 'y':
            print("Exiting without overwriting the file.")
            sys.exit()
        else:
            os.remove(filename)


def append_silence(tempfile, duration=1200):
    audio = AudioSegment.from_file(tempfile)
    # Create a silence segment
    silence = AudioSegment.silent(duration)
    # Append the silence segment to the audio
    combined = audio + silence
    # Save the combined audio back to file
    combined.export(tempfile, format="mp3")



def generate_metadata(files, author, title, chapter_titles):
    chap = 0
    start_time = 0
    with open("FFMETADATAFILE", "w") as file:
        file.write(";FFMETADATA1\n")
        file.write(f"ARTIST={author}\n")
        file.write(f"ALBUM={title}\n")
        file.write(
            "DESCRIPTION=Made with https://github.com/aedocw/epub2tts-edge\n")
        for file_name in files:
            duration = get_duration(file_name)
            file.write("[CHAPTER]\n")
            file.write("TIMEBASE=1/1000\n")
            file.write(f"START={start_time}\n")
            file.write(f"END={start_time + duration}\n")
            file.write(f"title={chapter_titles[chap]}\n")
            chap += 1
            start_time += duration


def get_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_milliseconds = len(audio)
    return duration_milliseconds


def make_m4b(files, sourcefile, speaker):
    filelist = "filelist.txt"
    basefile = sourcefile.replace(".txt", "")
    outputm4a = f"{basefile}-{speaker}.m4a"
    outputm4b = f"{basefile}-{speaker}.m4b"
    with open(filelist, "w") as f:
        for filename in files:
            filename = filename.replace("'", "'\\''")
            f.write(f"file '{filename}'\n")
    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        filelist,
        "-codec:a",
        "aac",
        "-b:a",
        "69k",
        "-f",
        "ipod",
        # "-filter:a",
        # f"atempo={speak_rate}",
        outputm4a,
    ]
    subprocess.run(ffmpeg_command)
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        outputm4a,
        "-i",
        "FFMETADATAFILE",
        "-map_metadata",
        "1",
        "-codec",
        "copy",
        outputm4b,
    ]
    subprocess.run(ffmpeg_command)
    os.remove(filelist)
    os.remove("FFMETADATAFILE")
    os.remove(outputm4a)
    for f in files:
        os.remove(f)
    return outputm4b


def add_cover(cover_img, filename):
    if os.path.isfile(cover_img):
        m4b = mp4.MP4(filename)
        cover_image = open(cover_img, "rb").read()
        m4b["covr"] = [mp4.MP4Cover(cover_image)]
        m4b.save()
    else:
        print(f"Cover image {cover_img} not found")


def run_edgespeak(sentence, speaker, filename, speak_rate):
    print(f"Running edge peak with filename {filename}")
    communicate = edge_tts.Communicate(sentence, speaker, rate=speak_rate)
    run_save(communicate, filename)


def run_save(communicate, filename):
    print(f"Saving {filename}...")
    asyncio.run(communicate.save(filename))

def read_book(book_contents, speaker, speak_rate="1.0"):
    segments = []
    for i, chapter in enumerate(book_contents, start=1):
        partname = f"part{i}.mp3"
        if os.path.isfile(partname):
            print(f"{partname} exists, skipping to next chapter")
            segments.append(partname)
        else:
            print(f"Chapter: {chapter['title']}\n")
            chapter_files = asyncio.run(parallel_edgespeak(
                [chapter['title']] + chapter["paragraphs"], [speaker] * (1 + len(chapter["paragraphs"])), speak_rate=speak_rate))
            
            # Combine chapter files
            append_silence(chapter_files[-1], 2800)
            combined = AudioSegment.empty()
            for file in chapter_files:
                combined += AudioSegment.from_mp3(file)
            combined.export(partname, format="mp3")
            for file in chapter_files:
                os.remove(file)
            
            segments.append(partname)
    return segments

async def parallel_edgespeak(texts, speakers, speak_rate="1.0", batch_size=5):
    semaphore = asyncio.Semaphore(_CONCURRENT_THREADS)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_THREADS) as executor:
        futures = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_speakers = speakers[i:i+batch_size]
            batch_sentences = []
            batch_filenames = []
            for text, speaker in zip(batch_texts, batch_speakers):
                sentences = sent_tokenize(text)
                batch_sentences.extend(sentences)
                batch_filenames.extend([f"temp_{j}.mp3" for j in range(len(batch_filenames), len(batch_filenames) + len(sentences))])
            async with semaphore:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(
                    executor, run_edgespeak_batch, batch_sentences, batch_speakers, batch_filenames, speak_rate)
                futures.append(future)

        for future in futures:
            batch_results = await future
            results.extend(batch_results)

    return results

async def parallel_edgespeak(texts, speakers, speak_rate="1.0", batch_size=5):
    semaphore = asyncio.Semaphore(_CONCURRENT_THREADS)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=_CONCURRENT_THREADS) as executor:
        for text, speaker in zip(texts, speakers):
            sentences = sent_tokenize(text)
            filenames = [f"temp_{j}.mp3" for j in range(len(results), len(results) + len(sentences))]
            async with semaphore:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(
                    executor, run_edgespeak_batch, sentences, [speaker] * len(sentences), filenames, speak_rate)
                batch_results = await future
                results.extend(batch_results)

    return results

def get_current_time():
  """Prints the current time in YYYY-MM-DD H:M:S format."""

  now = datetime.datetime.now()
  formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
  return formatted_time

def run_edgespeak_batch(sentences, speakers, filenames, speak_rate):
    batch_results = []
    for sentence, speaker, filename in zip(sentences, speakers, filenames):
        if os.path.isfile(filename):
            print(f"File {filename} already exists, skipping...")
            batch_results.append(filename)
            continue
        print(f"[{get_current_time()}] Communicating {filename}...")
        communicate = edge_tts.Communicate(sentence, speaker, rate=speak_rate, receive_timeout=10)
        print(f"[{get_current_time()}] Saving {filename}...")
        asyncio.run(communicate.save(filename))
        batch_results.append(filename)
    return batch_results


def main():
    parser = argparse.ArgumentParser(
        prog="epub2tts-edge",
        description="Read a text file to audiobook format",
    )
    parser.add_argument("sourcefile", type=str,
                        help="The epub or text file to process")
    parser.add_argument(
        "--speaker",
        type=str,
        nargs="?",
        const="en-US-AndrewNeural",
        default="en-US-AndrewNeural",
        help="Speaker to use (ex en-US-MichelleNeural)",
    )
    parser.add_argument("--speak_rate", type=str, default="+0%",
                        help="Speak rate to use. The default value is +0%")
    parser.add_argument(
        "--cover",
        type=str,
        help="jpg image to use for cover",
    )

    args = parser.parse_args()
    print(args)

    # If we get an epub, export that to txt file, then exit
    sourcefile = args.sourcefile
    if args.sourcefile.endswith(".epub"):
        book = epub.read_epub(args.sourcefile)
        sourcefile = export(book, args.sourcefile)

    book_contents, book_title, book_author, chapter_titles = get_book(
        sourcefile)
    files = read_book(book_contents, args.speaker, args.speak_rate)
    generate_metadata(files, book_author, book_title, chapter_titles)
    m4bfilename = make_m4b(files, args.sourcefile,
                           args.speaker)
    add_cover(args.cover, m4bfilename)


if __name__ == "__main__":
    main()

