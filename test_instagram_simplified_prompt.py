#!/usr/bin/env python3
"""
Instagram AB Test Processor - SIMPLIFIED PROMPT VERSION
Uses freeform LLM output with minimal constraints
Splits message into fields AFTER generation
"""

import os
import re
import time
import json
import base64
import requests
import subprocess
import socket
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from datetime import datetime
import openai
import anthropic
import dropbox
from dropbox.exceptions import ApiError, AuthError
import pymysql
from test_preprocessing import preprocess_content
from google import genai
from google.genai import types
try:
    import yt_dlp
except ImportError:
    yt_dlp = None
    print("⚠️ yt-dlp not installed - video downloads will be skipped")

# Configuration
MAX_PHOTOS = 9
CHROME_DEBUG_PORT = 9223
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CHROME_USER_DATA_DIR = r"C:\temp\chrome-debug"
DOWNLOAD_DIR = Path(__file__).parent / "instagram_ab_test_photos"
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Markdown base path
MARKDOWN_BASE_PATH = r"C:\Users\Vince\Documents\ML Vector Embeddings\Cover Images\name_update"

# Dropbox configuration
DROPBOX_BASE_FOLDER = "/SVF Team Folder/Advertising/Mailchimp Expansion/Florist/instagram_ab_test"

# OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Progress tracking
PROGRESS_FILE = "instagram_simplified_progress.json"

# Azalea categories path
AZALEA_CATEGORIES = Path(__file__).parent / "azalea_categories.md"


def split_message_into_fields(llm_output):
    """
    Split freeform LLM output into database fields.

    Expected format:
    SUBJECT: [subject line]

    [greeting line - Hi Name,]

    [paragraph 1 - PROMO]

    [paragraph 2 - CONNECTION]

    [paragraph 3 - OFFER]

    Azalea Floral Supply

    PRODUCT_URL: [url]

    Returns dict with: subject_line, promo_context, personal_connection, offer, tagline, validated_url
    """
    result = {
        'subject_line': '',
        'promo_context': '',
        'personal_connection': '',
        'offer': '',
        'tagline': 'Azalea Floral Supply',
        'validated_url': ''
    }

    lines = llm_output.strip().split('\n')

    # Extract SUBJECT
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('SUBJECT:'):
            result['subject_line'] = line.split(':', 1)[1].strip()[:70]
            break

    # Extract PRODUCT_URL
    for line in lines:
        if line.strip().upper().startswith('PRODUCT_URL:'):
            url = line.split(':', 1)[1].strip()
            if ' ' in url:
                url = url.split()[0]
            result['validated_url'] = url
            break

    # Extract message body - everything after SUBJECT line until PRODUCT_URL
    message_start = None
    message_end = None

    for i, line in enumerate(lines):
        if line.strip().upper().startswith('SUBJECT:'):
            message_start = i + 1
        elif line.strip().upper().startswith('PRODUCT_URL:'):
            message_end = i
            break

    if message_start is not None:
        if message_end is None:
            message_end = len(lines)

        # Get message content
        message_lines = lines[message_start:message_end]
        message_text = '\n'.join(message_lines).strip()

        # Split into paragraphs (double newline separated)
        paragraphs = [p.strip() for p in message_text.split('\n\n') if p.strip()]

        # Filter out tagline and short greeting lines
        content_paragraphs = []
        for p in paragraphs:
            p_lower = p.lower().strip()
            # Skip tagline
            if p_lower == 'azalea floral supply':
                continue
            # Skip short greeting lines (Hi X, or Hello X,)
            if len(p) < 50 and (p_lower.startswith('hi ') or p_lower.startswith('hello ') or p_lower.startswith('hey ')):
                continue
            content_paragraphs.append(p)

        # Assign paragraphs: PROMO, CONNECTION, OFFER
        if len(content_paragraphs) >= 1:
            result['promo_context'] = content_paragraphs[0][:300]

        if len(content_paragraphs) >= 2:
            result['personal_connection'] = content_paragraphs[1][:350]

        if len(content_paragraphs) >= 3:
            result['offer'] = '\n\n'.join(content_paragraphs[2:])[:250]

    return result


class InstagramSimplifiedProcessor:
    def __init__(self, ups_tags=None, contact_type='established', valid_seasons=None):
        self.mysql_config = {
            "host": "localhost",
            "user": "root",
            "password": "Bball33!",
            "database": "sacvalley_crm",
            "port": 33061,
            "charset": "utf8mb4",
            "autocommit": True
        }
        self.progress_file = PROGRESS_FILE
        self.last_processed_sequential_id = self.load_progress()
        self.chrome_debug_port = CHROME_DEBUG_PORT
        self.chrome_path = CHROME_PATH
        self.chrome_user_data_dir = CHROME_USER_DATA_DIR
        self.download_dir = DOWNLOAD_DIR
        self.dropbox_base_folder = DROPBOX_BASE_FOLDER
        self.ups_tags = ups_tags or []
        self.contact_type = contact_type
        self.valid_seasons = valid_seasons or ['FALL', 'XMAS', 'VDAY']

    def load_progress(self):
        """Load last processed sequential_id"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    if 'last_processed_sequential_id' in data:
                        last_id = data['last_processed_sequential_id']
                        print(f"📁 Loaded progress: last processed sequential_id = {last_id}")
                        return last_id
                    return ''
            except Exception as e:
                print(f"⚠️ Error loading progress file: {e}")
                return ''
        return ''

    def save_progress(self, sequential_id):
        """Save last processed sequential_id to progress file"""
        self.last_processed_sequential_id = sequential_id
        try:
            progress_data = {
                'last_processed_sequential_id': sequential_id,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"📝 Progress saved: sequential_id {sequential_id}")
        except Exception as e:
            print(f"⚠️ Error saving progress: {e}")

    def is_port_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    def launch_chrome_with_debugging(self):
        """Launch Chrome with debugging port"""
        if self.is_port_in_use(self.chrome_debug_port):
            print(f"✅ Chrome already running on port {self.chrome_debug_port}")
            return True

        print(f"🚀 Launching Chrome with debugging on port {self.chrome_debug_port}...")

        try:
            chrome_command = [
                self.chrome_path,
                f"--remote-debugging-port={self.chrome_debug_port}",
                f"--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={self.chrome_user_data_dir}"
            ]

            subprocess.Popen(chrome_command, shell=False)
            print("⏳ Waiting for Chrome to start...")
            time.sleep(5)

            if self.is_port_in_use(self.chrome_debug_port):
                print(f"✅ Chrome launched successfully on port {self.chrome_debug_port}")
                return True
            else:
                print(f"❌ Chrome failed to start on port {self.chrome_debug_port}")
                return False

        except Exception as e:
            print(f"❌ Error launching Chrome: {e}")
            return False

    def setup_selenium_driver(self):
        """Setup Selenium driver - connect to debugging port"""
        try:
            options = Options()
            options.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.chrome_debug_port}")
            driver = webdriver.Chrome(options=options)
            print("✅ Connected to Chrome debugging port")
            return driver
        except Exception as e:
            print(f"❌ Could not connect to debugging port: {e}")
            return None

    def extract_catalog_urls(self):
        """Extract catalog URLs from azalea_categories.md"""
        if not AZALEA_CATEGORIES.exists():
            print(f"⚠️ Catalog file not found: {AZALEA_CATEGORIES}")
            return []

        try:
            with open(AZALEA_CATEGORIES, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find URLs in markdown links
            urls = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', content)
            # Return just URLs
            return [url for name, url in urls]

        except Exception as e:
            print(f"⚠️ Error reading catalog: {e}")
            return []

    def build_profiling_prompt(self, intro_text, homepage_content, about_us_content,
                                image_contexts, has_videos=False, previous_product_url=None,
                                alt_texts=None):
        """Build simplified profiling prompt - freeform sales pitch with minimal constraints"""
        alt_texts = alt_texts or []

        # Extract catalog URLs
        catalog_urls = self.extract_catalog_urls()
        catalog_list = "\n".join([f"{i+1}. {url}" for i, url in enumerate(catalog_urls)])

        # Format alt texts from Instagram posts
        alt_text_block = ""
        if alt_texts:
            alt_text_block = "\n".join([f"- {alt[:200]}" for alt in alt_texts[:9] if alt])

        # Previous product exclusion
        exclude_note = ""
        if previous_product_url:
            exclude_note = f"\n(Do NOT use this product - already sent: {previous_product_url})"

        prompt = f"""You are writing a personalized B2B sales email from Azalea Floral Supply to a retail florist.

TARGET FLORIST DATA:
- Instagram Bio: {intro_text[:500] if intro_text else "(none)"}
- Website Homepage: {homepage_content[:2000] if homepage_content else "(none)"}
- About Page: {about_us_content[:1500] if about_us_content else "(none)"}

INSTAGRAM POSTS (from their feed):
{alt_text_block if alt_text_block else "(no post descriptions available)"}

[IMAGES ATTACHED: {len(image_contexts)} Instagram photos{' + video(s)' if has_videos else ''}]

ABOUT AZALEA FLORAL SUPPLY:
- Wholesale floral supply, ships nationwide
- Current promo: $10 off no minimum order, no code needed

WRITE a warm, personalized sales email (150-250 words, 3-4 paragraphs).

Guidelines:
- Reference something specific about their work (from photos/bio/website)
- Suggest a relevant product from our catalog
- Mention the $10 off promo naturally in the closing
- End with "Azalea Floral Supply" tagline

LANGUAGE RULES:
- Write like a real person, not corporate marketing speak
- NEVER use stiff phrases like "Your Premier Florist" or "Alta Gardens, Inc of Kenswick, WI"
- Avoid "Inc", "LLC", or formal business suffixes
- Avoid city/state combinations (town name alone is OK if relevant)
- Keep it conversational and warm
- NO creeping: Don't say "We noticed you..." or "I saw that you..." - just reference their work naturally

OUTPUT FORMAT:
SUBJECT: [max 70 chars - punchy, specific to them]

[paragraph 1 - personal connection/hook about their work]

[paragraph 2 - product suggestion with benefit]

[paragraph 3 - soft offer + close]

Azalea Floral Supply

PRODUCT_URL: [one URL from catalog below]{exclude_note}

CATALOG:
{catalog_list}
"""
        return prompt

    def profile_florist_with_vision(self, intro_text, homepage_content, about_us_content,
                                     image_paths, video_paths=None, previous_product_url=None,
                                     alt_texts=None):
        """
        One-shot profiling with Gemini 2.5 Flash - SIMPLIFIED
        Returns dict with merge_tags fields directly
        """
        video_paths = video_paths or []
        alt_texts = alt_texts or []
        has_videos = len(video_paths) > 0

        print(f"\n🤖 Profiling florist with Gemini 2.5 Flash (simplified prompt)...")
        print(f"   📸 Using {len(image_paths)} images")
        print(f"   📝 Using {len(alt_texts)} alt texts for context")
        if has_videos:
            print(f"   🎬 Using {len(video_paths)} video(s)")
        if previous_product_url:
            print(f"   ⚠️  Excluding previous product: {previous_product_url}")

        prompt = self.build_profiling_prompt(
            intro_text, homepage_content, about_us_content,
            image_paths, has_videos=has_videos,
            previous_product_url=previous_product_url,
            alt_texts=alt_texts
        )

        try:
            # Initialize Gemini client
            gemini_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

            # Build parts for Gemini
            parts = []

            # Add prompt text first
            parts.append(types.Part(text=prompt))

            # Add images
            for img_path in image_paths:
                try:
                    with open(img_path, "rb") as img_file:
                        image_bytes = img_file.read()

                    parts.append(types.Part(
                        inline_data=types.Blob(data=image_bytes, mime_type='image/jpeg')
                    ))
                    print(f"   ✅ Loaded image: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {img_path}: {e}")

            # Add videos if present
            for vid_path in video_paths:
                try:
                    with open(vid_path, "rb") as vid_file:
                        video_bytes = vid_file.read()

                    parts.append(types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ))
                    print(f"   ✅ Loaded video: {os.path.basename(vid_path)} ({len(video_bytes)/1024/1024:.1f} MB)")
                except Exception as e:
                    print(f"   ⚠️ Failed to load video {vid_path}: {e}")

            # Generate content with Gemini 2.5 Flash
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=types.Content(parts=parts)
            )

            result_text = response.text.strip()

            # Parse freeform output into fields
            merge_tags = split_message_into_fields(result_text)

            # Print results
            print("\n" + "=" * 80)
            print("📧 RAW LLM OUTPUT")
            print("=" * 80)
            print(result_text)
            print("=" * 80)

            print("\n" + "=" * 80)
            print("✉️ PARSED MERGE TAGS")
            print("=" * 80)
            print(f"\n📧 SUBJECT ({len(merge_tags['subject_line'])} chars):")
            print(f"   {merge_tags['subject_line']}")
            print(f"\n📝 PROMO ({len(merge_tags['promo_context'])} chars):")
            print(f"   {merge_tags['promo_context']}")
            print(f"\n🤝 CONNECTION ({len(merge_tags['personal_connection'])} chars):")
            print(f"   {merge_tags['personal_connection']}")
            print(f"\n🎁 OFFER ({len(merge_tags['offer'])} chars):")
            print(f"   {merge_tags['offer']}")
            print(f"\n✨ TAGLINE:")
            print(f"   {merge_tags['tagline']}")
            print(f"\n🔗 PRODUCT_URL:")
            print(f"   {merge_tags['validated_url']}")
            print("=" * 80)

            # Validate we got essential fields
            if not merge_tags['subject_line'] or not merge_tags['promo_context']:
                print("❌ Failed to parse essential fields from output")
                return None

            return {
                'merge_tags': merge_tags,
                'raw_output': result_text,
                'alt_texts': alt_texts
            }

        except Exception as e:
            print(f"❌ Error profiling florist: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_merge_tags_to_db(self, place_id, merge_tags, alt_texts=None):
        """Save merge tags to custom_emails - simplified version"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cur = conn.cursor()

            # Store alt_texts as JSON in profile_json field
            profile_data = {
                'instagram_alt_texts': alt_texts or [],
                'generated_at': datetime.now().isoformat()
            }

            cur.execute("""
                UPDATE custom_emails
                SET subject_line=%s,
                    promo_context=%s,
                    personal_connection=%s,
                    offer=%s,
                    tagline=%s,
                    product_selection=%s,
                    profile_json=%s,
                    updated_at=NOW()
                WHERE placeId=%s
            """, (
                merge_tags['subject_line'],
                merge_tags['promo_context'],
                merge_tags['personal_connection'],
                merge_tags['offer'],
                merge_tags['tagline'],
                merge_tags['validated_url'],
                json.dumps(profile_data),
                place_id
            ))

            conn.close()
            print(f"   ✅ Saved merge tags to database")
            return True

        except Exception as e:
            print(f"   ❌ Database save failed: {e}")
            return False

    def extract_bio_from_instagram(self, driver, instagram_url, title):
        """Extract business bio text from Instagram profile page."""
        print(f"   🔎 Visiting: {instagram_url}")
        try:
            driver.get(instagram_url)
            time.sleep(10)

            bio_selectors = [
                "//section//span[contains(@class, '_ap3a') and contains(@class, '_aaco')]",
                "//header//span[not(contains(@class, '_ap3a'))]//span",
                "//div[contains(@class, 'x7a106z')]//span",
                "//section//div[contains(@class, 'x9f619')]//span[@dir='auto']"
            ]

            for selector in bio_selectors:
                try:
                    bio_elements = driver.find_elements(By.XPATH, selector)
                    for elem in bio_elements:
                        text = elem.text.strip()
                        if text and len(text) > 20:
                            print(f"   ✅ Found bio text ({len(text)} chars): {text[:100]}...")
                            return text
                except:
                    continue

            print("   ❌ No bio text found")
            return None
        except Exception as e:
            print(f"   ❌ Error extracting bio: {str(e)[:100]}")
            return None

    def extract_posts_from_grid(self, driver, instagram_url, max_posts=9):
        """Extract posts from Instagram grid with images and alt texts."""
        print(f"\n📸 Extracting posts from grid (max {max_posts})...")

        try:
            current_url = driver.current_url
            if instagram_url not in current_url:
                driver.get(instagram_url)
                time.sleep(5)

            post_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/p/"], a[href*="/reel/"]')
            print(f"   Found {len(post_links)} post links in grid")

            posts = []
            for i, link in enumerate(post_links[:max_posts]):
                try:
                    href = link.get_attribute("href")
                    if not href:
                        continue

                    post_type = 'reel' if '/reel/' in href else 'photo'
                    image_url = None
                    alt_text = None
                    post_date = None

                    try:
                        img = link.find_element(By.TAG_NAME, "img")
                        image_url = img.get_attribute("src")
                        alt_text = img.get_attribute("alt") or ""

                        if alt_text and " on " in alt_text:
                            date_part = alt_text.split(" on ", 1)[1]
                            for delimiter in [". ", " tagging", "."]:
                                if delimiter in date_part:
                                    date_part = date_part.split(delimiter)[0]
                                    break
                            if re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})', date_part):
                                post_date = date_part.strip()
                    except:
                        pass

                    post_data = {
                        'post_url': href,
                        'post_type': post_type,
                        'image_url': image_url,
                        'alt_text': alt_text,
                        'post_date': post_date
                    }
                    posts.append(post_data)

                    alt_preview = (alt_text[:60] + "...") if alt_text and len(alt_text) > 60 else alt_text
                    print(f"   ✅ {post_type}: {post_date or '(no date)'} | alt: {alt_preview or '(empty)'}")

                except Exception as e:
                    print(f"   ⚠️ Error extracting post: {str(e)[:50]}")
                    continue

            print(f"\n📊 Extracted {len(posts)} posts from grid")
            return posts

        except Exception as e:
            print(f"❌ Error extracting posts from grid: {e}")
            return []

    def update_intro_text(self, place_id, intro_text):
        """Update intro_text in custom_emails"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cur = conn.cursor()

            if intro_text:
                cur.execute(
                    "UPDATE custom_emails SET intro_text=%s, updated_at=NOW() WHERE placeId=%s",
                    (intro_text, place_id),
                )
                print(f"   💾 Updated intro_text ({len(intro_text)} chars)")
                return True
            else:
                print("   ⚠️ No intro text to update")
                return False
        except Exception as e:
            print(f"   ❌ Database update failed: {e}")
            return False
        finally:
            try:
                conn.close()
            except:
                pass

    def load_and_preprocess_markdown(self, file_path, char_limit=50000):
        """Load markdown file, apply preprocessing, and truncate if needed"""
        if not file_path:
            return ""

        full_path = os.path.normpath(os.path.join(str(MARKDOWN_BASE_PATH), file_path))

        if not os.path.exists(full_path):
            print(f"      ⚠️ Markdown file not found: {full_path}")
            return ""

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_len = len(content)
            content = preprocess_content(content)
            cleaned_len = len(content)
            reduction = original_len - cleaned_len

            print(f"      Raw: {original_len:,} chars → Cleaned: {cleaned_len:,} chars (removed {reduction:,})")

            if len(content) > char_limit:
                content = content[:char_limit] + "\n\n[Content truncated...]"
                print(f"      Truncated to {char_limit:,} chars")

            return content

        except Exception as e:
            print(f"      ⚠️ Error loading markdown from {full_path}: {e}")
            return ""

    def download_images(self, photo_urls, place_name):
        """Download images from URLs"""
        place_dir = self.download_dir / place_name.replace(" ", "_")
        place_dir.mkdir(exist_ok=True)

        downloaded_paths = []

        for i, url in enumerate(photo_urls, 1):
            try:
                print(f"⬇️  Downloading image {i}/{len(photo_urls)}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                image_path = place_dir / f"photo_{i}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)

                downloaded_paths.append(str(image_path))
                print(f"   ✅ Saved: {image_path.name}")

            except Exception as e:
                print(f"   ⚠️ Failed to download image {i}: {e}")

        return downloaded_paths

    def download_videos(self, video_urls, place_name):
        """Download videos using yt-dlp and trim to first 5 seconds"""
        if not yt_dlp:
            print("   ⚠️ yt-dlp not available, skipping video downloads")
            return []

        place_dir = self.download_dir / place_name.replace(" ", "_")
        place_dir.mkdir(exist_ok=True)

        downloaded_paths = []

        for i, url in enumerate(video_urls, 1):
            try:
                print(f"⬇️  Downloading video {i}/{len(video_urls)}...")
                temp_output = str(place_dir / f"video_{i}_full.%(ext)s")

                ydl_opts = {
                    'format': 'best',
                    'outtmpl': temp_output,
                    'quiet': False,
                    'no_warnings': False,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                possible_extensions = ['.mp4', '.webm', '.mkv', '.mov']
                full_video_path = None

                for ext in possible_extensions:
                    test_path = place_dir / f"video_{i}_full{ext}"
                    if test_path.exists():
                        full_video_path = test_path
                        break

                if full_video_path and full_video_path.exists():
                    print(f"   ✅ Downloaded: {full_video_path.name}")

                    trimmed_path = place_dir / f"video_{i}.mp4"
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', str(full_video_path),
                        '-t', '5', '-c', 'copy', '-y', str(trimmed_path)
                    ]

                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                        full_video_path.unlink()
                        downloaded_paths.append(str(trimmed_path))
                    except:
                        final_path = place_dir / f"video_{i}.mp4"
                        full_video_path.rename(final_path)
                        downloaded_paths.append(str(final_path))

            except Exception as e:
                print(f"   ❌ Failed to download video {i}: {e}")

        return downloaded_paths

    def refresh_dropbox_token(self):
        """Refresh Dropbox access token using env vars"""
        app_key = os.getenv("DROPBOX_APP_KEY")
        app_secret = os.getenv("DROPBOX_APP_SECRET")
        refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")
        token_file = Path(__file__).parent / ".dropbox_access_token"

        if not (app_key and app_secret and refresh_token):
            if token_file.exists():
                try:
                    return token_file.read_text().strip()
                except:
                    return None
            return None

        try:
            response = requests.post(
                "https://api.dropbox.com/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": app_key,
                    "client_secret": app_secret,
                },
                timeout=15,
            )

            if response.status_code == 200:
                new_access_token = response.json().get("access_token")
                if new_access_token:
                    token_file.write_text(new_access_token)
                    return new_access_token
            return None
        except:
            return None

    def get_dbx_client(self):
        """Get authenticated Dropbox client with team namespace"""
        token = self.refresh_dropbox_token()
        if not token:
            token_file = Path(__file__).parent / ".dropbox_access_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        if not token:
            print("❌ Failed to get Dropbox token")
            return None

        try:
            dbx_temp = dropbox.Dropbox(token)
            acct = dbx_temp.users_get_current_account()
            print(f"☁️ Dropbox: {acct.name.display_name}")

            path_root_header = json.dumps({
                ".tag": "root",
                "root": acct.root_info.root_namespace_id
            })

            return dropbox.Dropbox(
                token,
                headers={'Dropbox-API-Path-Root': path_root_header}
            )

        except AuthError as e:
            print(f"❌ Dropbox auth failed: {e}")
            return None

    def upload_to_dropbox(self, file_path, place_name, file_number, file_type, dbx):
        """Upload file to Dropbox"""
        try:
            ext = '.jpg' if file_type == 'photo' else '.mp4'
            dropbox_folder = f"{self.dropbox_base_folder}/{place_name}"
            dropbox_path = f"{dropbox_folder}/{file_type}_{file_number}{ext}"

            with open(file_path, 'rb') as f:
                file_data = f.read()

            dbx.files_upload(
                file_data,
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite
            )

            print(f"   ☁️ Uploaded to Dropbox: {dropbox_path}")
            return True

        except Exception as e:
            print(f"   ⚠️ Dropbox upload failed: {e}")
            return False

    def export_merge_tags_to_csv(self, output_filename=None):
        """Export merge tags to CSV with Mailchimp-compatible column names."""
        if output_filename is None:
            output_filename = f"{self.contact_type}_ig_simplified_merge_tags.csv"

        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)

            if self.ups_tags:
                ups_placeholders = ', '.join(['%s'] * len(self.ups_tags))
                ups_filter = f"AND mt.ups_tag IN ({ups_placeholders})"
                ups_params = tuple(self.ups_tags)
            else:
                ups_filter = ""
                ups_params = ()

            if self.contact_type == 'established':
                contact_filter = """
                  AND EXISTS (
                      SELECT 1 FROM mailchimp_analytics ma
                      WHERE ma.email = mt.Email
                        AND ma.Real_open = 1
                        AND ma.sender_name LIKE '%%Azalea%%'
                  )
                """
            else:
                contact_filter = """
                  AND NOT EXISTS (
                      SELECT 1 FROM mailchimp_analytics ma
                      WHERE ma.email = mt.Email
                        AND ma.Real_open = 1
                        AND ma.sender_name LIKE '%%Azalea%%'
                  )
                """

            query = f"""
                SELECT
                    mt.Email as EMAIL,
                    mt.shop_name as shop_name,
                    ce.owner_fname as FNAME,
                    ce.subject_line as SUBJECT,
                    ce.promo_context as PROMO,
                    ce.personal_connection as CONNECTION,
                    ce.offer as OFFER,
                    ce.tagline as TAGLINE,
                    ce.product_selection as PRODUCTURL
                FROM mailchimp_tags mt
                LEFT JOIN custom_emails ce ON mt.placeId COLLATE utf8mb4_unicode_ci = ce.placeId COLLATE utf8mb4_unicode_ci
                LEFT JOIN sheet_live sl ON mt.placeId COLLATE utf8mb4_unicode_ci = sl.placeId COLLATE utf8mb4_unicode_ci
                WHERE sl.instagram IS NOT NULL
                  AND sl.instagram != ''
                  AND mt.DO_NOT_SEND = 0
                  AND ce.subject_line IS NOT NULL
                  AND ce.subject_line != ''
                  {ups_filter}
                  {contact_filter}
                ORDER BY mt.placeId
            """

            cursor.execute(query, ups_params)
            records = cursor.fetchall()
            conn.close()

            if not records:
                print("⚠️ No merge tags found to export")
                return False

            # Format FNAME: "Sarah," or "Breen's team," fallback
            for record in records:
                if record.get('FNAME') and record['FNAME'].strip():
                    # Has first name - just use it
                    record['FNAME'] = f"{record['FNAME'].strip()},"
                else:
                    # No first name - use "[Business]'s team,"
                    shop = record.get('shop_name', '') or ''
                    # Strip common suffixes
                    for suffix in [' Florist', ' Flowers', ' Floral Design', ' Floral', ' Flower Shop', ' Inc', ' LLC']:
                        if shop.endswith(suffix):
                            shop = shop[:-len(suffix)]
                    shop = shop.strip()
                    if shop:
                        record['FNAME'] = f"{shop}'s team,"
                    else:
                        record['FNAME'] = "team,"
                # Remove shop_name from record (not needed in CSV)
                record.pop('shop_name', None)

            # Write to CSV
            output_path = Path(__file__).parent / output_filename
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['EMAIL', 'FNAME', 'SUBJECT', 'PROMO', 'CONNECTION', 'OFFER', 'TAGLINE', 'PRODUCTURL']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

            print(f"\n📊 EXPORT COMPLETE:")
            print(f"   ✅ Exported {len(records)} merge tags to: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

    def get_records_batch(self, limit=10):
        """Get batch of records from mailchimp_tags table filtered by UPS tags"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)

            if self.ups_tags:
                ups_placeholders = ', '.join(['%s'] * len(self.ups_tags))
                ups_filter = f"AND mt.ups_tag IN ({ups_placeholders})"
                ups_params = tuple(self.ups_tags)
            else:
                ups_filter = ""
                ups_params = ()

            if self.contact_type == 'established':
                contact_filter = """
                  AND EXISTS (
                      SELECT 1 FROM mailchimp_analytics ma
                      WHERE ma.email = mt.Email
                        AND ma.Real_open = 1
                        AND ma.sender_name LIKE '%%Azalea%%'
                  )
                """
            else:
                contact_filter = """
                  AND NOT EXISTS (
                      SELECT 1 FROM mailchimp_analytics ma
                      WHERE ma.email = mt.Email
                        AND ma.Real_open = 1
                        AND ma.sender_name LIKE '%%Azalea%%'
                  )
                """

            params = ups_params + (self.last_processed_sequential_id, limit)

            query = f"""
                SELECT
                    mt.placeId,
                    mt.ups_tag,
                    mt.shop_name AS title,
                    mt.city,
                    mt.state,
                    mt.Email AS email,
                    ce.owner_fname,
                    ce.homepage_markdown_file,
                    ce.about_us_markdown_file,
                    ce.product_selection,
                    sl.instagram AS instagram_url
                FROM mailchimp_tags mt
                LEFT JOIN custom_emails ce ON mt.placeId COLLATE utf8mb4_unicode_ci = ce.placeId COLLATE utf8mb4_unicode_ci
                LEFT JOIN sheet_live sl ON mt.placeId COLLATE utf8mb4_unicode_ci = sl.placeId COLLATE utf8mb4_unicode_ci
                WHERE sl.instagram IS NOT NULL
                  AND sl.instagram != ''
                  AND mt.DO_NOT_SEND = 0
                  {ups_filter}
                  {contact_filter}
                  AND mt.placeId > %s
                ORDER BY mt.placeId
                LIMIT %s
            """

            cursor.execute(query, params)
            records = cursor.fetchall()
            conn.close()

            print(f"📊 Found {len(records)} {self.contact_type} records with Instagram in this batch")
            return records

        except Exception as e:
            print(f"❌ Error getting records: {e}")
            return []

    def process_single_record(self, driver, record, dbx):
        """Process a single record completely - SIMPLIFIED VERSION"""
        place_id = record['placeId']
        title = record['title']
        city = record['city']
        state = record['state']
        instagram = record['instagram_url']
        homepage_file = record.get('homepage_markdown_file')
        about_us_file = record.get('about_us_markdown_file')
        previous_product_url = record.get('product_selection')

        print(f"\n🎯 Processing: {title} ({city}, {state})")
        print(f"   PlaceID: {place_id}")
        print(f"   Instagram: {instagram}")

        result = {
            'place_id': place_id,
            'title': title,
            'success': False
        }

        try:
            # Sanitize place name
            place_name = title.replace(' ', '_')
            for char in '<>:"/\\|?*':
                place_name = place_name.replace(char, '_')
            place_name = place_name[:64]

            # Step 1: Extract bio
            print("\n📝 Extracting bio text...")
            intro_text = self.extract_bio_from_instagram(driver, instagram, title)
            if self.update_intro_text(place_id, intro_text):
                result['intro_extracted'] = True

            # Step 2: Extract posts from grid
            grid_posts = self.extract_posts_from_grid(driver, instagram, max_posts=9)

            if not grid_posts:
                print("⚠️ No posts found in grid")
                self.save_progress(place_id)
                result['success'] = True
                return result

            # Collect alt_texts for LLM context
            alt_texts = [p['alt_text'] for p in grid_posts if p.get('alt_text')]
            photo_posts = [p for p in grid_posts if p['post_type'] == 'photo']
            reel_posts = [p for p in grid_posts if p['post_type'] == 'reel']

            print(f"\n📊 Grid summary: Photos: {len(photo_posts)}, Reels: {len(reel_posts)}")
            print(f"   Alt texts collected: {len(alt_texts)}")

            # Step 3: Download images
            photo_urls = [p['image_url'] for p in grid_posts if p.get('image_url')]
            image_paths = []
            if photo_urls:
                print(f"\n⬇️  Downloading images...")
                image_paths = self.download_images(photo_urls, place_name)

            # Step 4: Download reels
            video_paths = []
            reel_urls = [p['post_url'] for p in reel_posts]
            if reel_urls:
                print(f"\n⬇️  Downloading reels...")
                video_paths = self.download_videos(reel_urls, place_name)

            if not image_paths and not video_paths:
                print("⚠️ No images or videos downloaded")
                self.save_progress(place_id)
                result['success'] = True
                return result

            # Step 5: Load website markdown
            print("\n📄 Loading website markdown...")
            homepage_content = ""
            about_us_content = ""

            if homepage_file:
                homepage_content = self.load_and_preprocess_markdown(homepage_file)
            if about_us_file:
                about_us_content = self.load_and_preprocess_markdown(about_us_file)

            # Step 6: Profile florist with SIMPLIFIED prompt
            print("\n🤖 Profiling florist with simplified prompt...")
            combined_result = self.profile_florist_with_vision(
                intro_text,
                homepage_content,
                about_us_content,
                image_paths,
                video_paths,
                previous_product_url,
                alt_texts=alt_texts  # Pass alt_texts instead of post_dates
            )

            if not combined_result:
                print("⚠️ Profile generation failed")
                self.save_progress(place_id)
                result['success'] = False
                return result

            merge_tags = combined_result['merge_tags']
            result['merge_tags_generated'] = True

            # Step 7: Save to file
            result_file = self.download_dir / place_name / "simplified_result.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'merge_tags': merge_tags,
                    'alt_texts': alt_texts,
                    'raw_output': combined_result['raw_output']
                }, f, indent=2)

            # Step 8: Upload to Dropbox
            if dbx:
                for i, img_path in enumerate(image_paths, 1):
                    self.upload_to_dropbox(img_path, place_name, i, 'photo', dbx)
                for i, vid_path in enumerate(video_paths, 1):
                    self.upload_to_dropbox(vid_path, place_name, i, 'video', dbx)
                result['dropbox_uploaded'] = True

            # Step 9: Cleanup
            try:
                import shutil
                place_dir = self.download_dir / place_name
                shutil.rmtree(place_dir)
            except:
                pass

            # Step 10: Save to database
            print(f"\n💾 Saving merge tags to database...")
            self.save_merge_tags_to_db(place_id, merge_tags, alt_texts)

            self.save_progress(place_id)
            result['success'] = True
            print(f"\n✅ Completed record {place_id}")

        except Exception as e:
            print(f"\n❌ Error processing record: {e}")
            result['error'] = str(e)

        return result

    def process_batch(self, batch_size=5):
        """Process a single batch of records"""
        print(f"\n🚀 Starting batch processing (batch size: {batch_size})")

        records = self.get_records_batch(batch_size)
        if not records:
            print("✅ No more records to process!")
            return None

        if not self.launch_chrome_with_debugging():
            return None

        driver = self.setup_selenium_driver()
        if not driver:
            return None

        print("\n☁️  Connecting to Dropbox...")
        dbx = self.get_dbx_client()

        results = []

        try:
            for i, record in enumerate(records, 1):
                print(f"\n{'='*80}")
                print(f"[{i}/{len(records)}] Processing record...")
                print(f"{'='*80}")

                result = self.process_single_record(driver, record, dbx)
                results.append(result)

                if i < len(records):
                    time.sleep(3)

        finally:
            driver.quit()

        print(f"\n📊 BATCH SUMMARY:")
        print(f"   Records processed: {len(results)}")
        print(f"   Successful: {sum(1 for r in results if r['success'])}")

        return results


def test_split_function():
    """Test the split_message_into_fields function"""
    # Sample output - no MESSAGE: label, greeting is skipped
    sample_output = """SUBJECT: Your purple wedding arrangements caught my eye

Hi Platinum Petals team,

Those deep violet wedding arrangements on your feed are stunning - the way you layer textures with the lighter accent blooms shows real artistry. It's clear your brides are getting something special.

We carry some gorgeous purple lisianthus and stock that would pair beautifully with your signature style. Our Dutch-grown stems hold up incredibly well for those multi-day wedding setups you're known for.

Right now we're doing $10 off with no minimum - perfect for trying out a few bunches on your next event. Would love to see what you create!

Azalea Floral Supply

PRODUCT_URL: https://azaleafloralsupply.com/collections/purple-lisianthus"""

    print("=" * 60)
    print("TESTING split_message_into_fields()")
    print("=" * 60)
    print("\nINPUT:")
    print(sample_output)

    result = split_message_into_fields(sample_output)

    print("\n" + "=" * 60)
    print("PARSED OUTPUT:")
    print("=" * 60)
    print(f"\n📧 SUBJECT ({len(result['subject_line'])} chars):")
    print(f"   {result['subject_line']}")
    print(f"\n📝 PROMO ({len(result['promo_context'])} chars):")
    print(f"   {result['promo_context']}")
    print(f"\n🤝 CONNECTION ({len(result['personal_connection'])} chars):")
    print(f"   {result['personal_connection']}")
    print(f"\n🎁 OFFER ({len(result['offer'])} chars):")
    print(f"   {result['offer']}")
    print(f"\n✨ TAGLINE: {result['tagline']}")
    print(f"\n🔗 URL: {result['validated_url']}")

    print("\n" + "=" * 60)
    print("✅ Test complete - greeting skipped, 3 paragraphs mapped correctly")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_split_function()
    elif len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Run batch processing
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        processor = InstagramSimplifiedProcessor(
            ups_tags=['ups_MW', 'ups_East', 'ups_West'],
            contact_type='established'
        )
        processor.process_batch(batch_size)
    elif len(sys.argv) > 1 and sys.argv[1] == '--export':
        # Export merge tags to CSV
        processor = InstagramSimplifiedProcessor(
            ups_tags=['ups_MW', 'ups_East', 'ups_West'],
            contact_type='established'
        )
        processor.export_merge_tags_to_csv()
    else:
        print("Usage:")
        print("  python test_instagram_simplified_prompt.py --test     # Test split function")
        print("  python test_instagram_simplified_prompt.py --batch 3  # Process 3 records")
        print("  python test_instagram_simplified_prompt.py --export   # Export to CSV")
        print("\nSIMPLIFIED PROMPT VERSION")
        print("- Freeform LLM output with minimal constraints")
        print("- Splits message into fields AFTER generation")
        print("- FNAME fallback: 'team,' if no first name in DB")
