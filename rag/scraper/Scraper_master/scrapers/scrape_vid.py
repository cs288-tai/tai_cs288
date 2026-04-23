import os
from urllib.parse import parse_qs, urlparse

import requests
import yaml
import yt_dlp
from bs4 import BeautifulSoup

from scraper.Scraper_master.scrapers.base_scraper import BaseScraper
from scraper.Scraper_master.utils.file_utils import *


class VideoScraper(BaseScraper):
    def scrape(self, url, driver, task_folder_path):
        # Set base folder for all downloads
        os.makedirs(task_folder_path, exist_ok=True)
        self._download_video(url, task_folder_path)
        return []

    def _extract_youtube_links(self, soup):
        """Extracts YouTube video and playlist links from anchor tags in HTML."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            url = a_tag["href"]
            if self._is_youtube_url(url):
                links.append(url)
        return links

    def _is_youtube_url(self, url):
        """Checks if a URL belongs to YouTube."""
        parsed_url = urlparse(url)
        return parsed_url.netloc in [
            "www.youtube.com",
            "youtube.com",
            "m.youtube.com",
            "youtu.be",
        ]

    def _is_unviewable_playlist(self, url):
        """
        Check if URL is an unviewable playlist type (RD playlists, RDMM, etc.).
        These are auto-generated recommendation playlists that cannot be downloaded.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        playlist_id = query_params.get('list', [''])[0]

        # Skip RD (radio/recommendations) and similar auto-generated playlists
        unviewable_prefixes = ['RD', 'RDMM', 'RDEM', 'RDCMUC', 'RDAMVM', 'RDTMVM']
        return any(playlist_id.startswith(prefix) for prefix in unviewable_prefixes)

    def _download_video(self, url, folder):
        """Downloads a single video and saves metadata."""
        # Skip unviewable playlist types
        if self._is_unviewable_playlist(url):
            print(f"Skipping unviewable playlist: {url}")
            return

        # Limit filename length to prevent filesystem errors (max 100 chars for title)
        outtmpl = os.path.join(folder, "./%(playlist_title&{}|).100B/%(playlist_index&{}-|)s%(title).100B.%(ext)s")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            # "cookiefile": "/home/bot/bot/yk/YK_final/www.youtube.com_cookies.txt",
            "cookiesfrombrowser": ('chrome',),
            "outtmpl": outtmpl,
            "ignoreerrors": True,
            "format": "bv*+ba/b/best",

            # Network retry settings for HTTP 500 and connection errors
            "retries": 10,
            "fragment_retries": 10,
            "skip_unavailable_fragments": True,
            "abort_on_unavailable_fragments": False,

            # Rate limiting to prevent resource exhaustion
            "ratelimit": 5242880,  # 5MB/s limit
            "sleep_interval": 5,
            "max_sleep_interval": 10,

            # Handle partial downloads
            "noresizebuffer": True,
            "http_chunk_size": 10485760,  # 10MB chunks
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(url, download=True)
        if not video_info:
            print(f"Skipping {url}, unable to retrieve info.")
            return
        if video_info.get('_type') == 'playlist':
            playlist_url = video_info.get('webpage_url')
            entries = video_info.get('entries') or []
            for entry in entries:
                if not entry:
                    continue
                requested = entry.get('requested_downloads')
                if not requested:
                    continue
                first = requested[0] or {}
                filepath = first.get('filepath')
                if not filepath:
                    continue
                video_url = entry.get('original_url') or entry.get('url')
                additional_metadata = {
                    'playlist_url': playlist_url,
                }
                self._save_metadata(filepath, video_url, additional_metadata)
        else:
            # For single video
            video_url = video_info.get('original_url') or video_info.get('url')
            requested = video_info.get('requested_downloads') or []
            if not requested:
                print(f"Skipping {url}, no download produced.")
                return
            filepath = (requested[0] or {}).get('filepath')
            if not filepath:
                print(f"Skipping {url}, no filepath in download info.")
                return
            self._save_metadata(filepath, video_url)


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=1P2UgdAWwYg&list=PL6BsET-8jgYXTuSlJNYQS740YMCRHT79g&ab_channel=JohnDeNero"
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    print("list" in query_params and "playlist" in parsed_url.path)
