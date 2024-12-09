import yt_dlp
import os
import sys
import json
import time

fpv_keywords = ['fpv', 'dji avata']
drone_keywords = ['drone', 'dji', 'fpv']
skip_words = ['unbox', 'piyush', 'drone catches', 'glue gun', 'humne', 'gets attacked', 'drone attack', 'sonic drone',
              'zz kid', 'merekam', 'scary video', 'drone show', 'Arcade Craniacs', '.exe', 'Stromedy', 'zombie',
              'restore', 'Jester', 'drone strike', 'Mark Rober', 'Plasmonix', 'Sourav Joshi Vlogs', 'lego', 'MR. INDIAN HACKER',
              'horror', 'ketakutan', 'Mikael TubeHD', 'TRT \u00c7ocuk', 'kurulacak', 'penampakan', 'The RawKnee Games',
              'NIkku Vlogz', 'girgaya', 'chinu', 'military', 'surveillance', 'pakai', 'Yudist Ardhana', 'Technical Indian with fun',
              'menangkap', 'nampak', 'ZZ Kids TV', 'soldier', 'Bharat Tak', 'Sannu Kumar', 'lahari', 'Mikael Family', 'Smart rohit gadget',
              'attacked by', 'Josh Reid', 'terekam', 'kentang', 'sumur maut', 'Frost Diamond', 'Fatih Can Aytan', 'apex legends',
              'Golden TV - Funny Hindi Comedy Videos', 'hindi comedy', 'The Squeezed Lemon', 'no one would have believed it',
              'Wonders of the World', 'carwow', 'no one would believe it', 'Max TV', 'cartoon', 'for kids', 'pendaki', 'andrea ramadhan',
              'tornado', 'corpse', 'aerial tale', 'muthumalai', 'Videogyan Kids Shows - Toddler Learning Videos', 'WasimOP', 'nangis',
              'Zefanya Oyanio', 'deadly fire', 'oblivion', 'darussalam', 'Kang Adink', 'shoot down', 'collapse', 'hamas', 'aftermath',
              'NewsFirst Kannada', 'drone fails', 'FailArmy', 'YaQurban', 'pashto', 'camping', 'Harmen Hoek', 'slaughter', 'weapon',
              'JerryRigEverything', 'toys', 'Majedar Review', ]


def get_video_stat(metadata):
    def contain(xs, keywords):
        return any(keyword in x for keyword in keywords for x in xs)

    def has_skip_language(text):
        import re
        from langdetect import detect

        def remove_emojis(text):
            # This pattern matches emojis in the text
            emoji_pattern = re.compile(
                "["u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"  # other symbols
                u"\U000024C2-\U0001F251"  # enclosed characters
                "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
        try:
            text_without_emojis = remove_emojis(text)
            if text_without_emojis.strip() == "":
                return False  # Only emojis or empty string

            detected_language = detect(text_without_emojis)
            # skip south asian and middle eastern languages
            if detected_language in ['hi', 'bn', 'ur', 'ar', 'fa', 'he', 'th', 'ja', 'ko', 'zh']:
                return True
            else:
                return False
        except Exception as e:
            # Handle the case where the detection fails (e.g., the text is too short)
            return False

    like_count = metadata.get('like_count', None)
    comment_count = metadata.get('comment_count', None)
    is_fpv = contain([metadata['title'].lower(), metadata['description'].lower()],
                     fpv_keywords)
    is_drone = contain([metadata['title'].lower(), metadata['description'].lower()],
                       drone_keywords)
    has_skip_words = contain([metadata['title'].lower(), metadata['description'].lower(), metadata.get('channel', '')],
                             skip_words)
    width, height = map(lambda x: int(x),
                        metadata['resolution'].split('x'))
    is_landscape = width / height > 1
    video_stat = {
        'title': metadata['title'],
        'view_count': metadata['view_count'],
        'like_count': like_count if like_count is not None else 0,
        'comment_count': comment_count if comment_count is not None else 0,
        'duration': metadata['duration'],
        'is_fpv': is_fpv,
        'has_skip_words': has_skip_words,
        'is_drone': is_drone,
        'is_landscape': is_landscape,
        # 'has_skip_language': has_skip_language(metadata['title'].lower()),
    }
    return video_stat


def download_youtube_video(download_path, video_id, video_resolution=1920, max_video_length=None, quiet=True):
    # Replace with the desired video URL
    video_url = f'http://www.youtube.com/watch?v={video_id}'

    os.makedirs(f'{download_path}/{video_id}', exist_ok=True)

    t0 = time.time()
    if quiet:
        # Open a file for writing output
        # Save the current stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Redirect stdout and stderr to os.devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    # Define youtube-dlp options
    ydl_opts = {
        # "noprogress": quiet,
        # "quiet": quiet,
        'format_sort': ['vcodec:h264', 'res', 'acodec:m4a'],
        # make sure the longer side is less than or equal to video_resolution
        'format': f'bestvideo[ext=mp4][vcodec!*=hdr][width<={video_resolution}][height<={video_resolution}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        # 'match_filter': yt_dlp.utils.match_filter_func(f"duration < {max_video_length}"),
        'outtmpl': f'{download_path}/{video_id}/video.%(ext)s',
        # 'writedescription': True,  # Write the video description to a .description file
        # 'writeinfojson': True,  # Write the video metadata to a .info.json file
        'writesubtitles': True,   # Write subtitles
        'writeautomaticsub': True  # Write auto-generated subtitles
    }
    if max_video_length is not None:
        ydl_opts['match_filter'] = yt_dlp.utils.match_filter_func(
            f"duration < {max_video_length}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(video_url)
        with open(f'{download_path}/{video_id}/data.json', 'w') as fp:
            for key in ['formats', 'thumbnails', 'automatic_captions', 'subtitles', 'heatmap', 'requested_downloads', 'requested_formats']:
                del video_info[key]
            json.dump(video_info, fp, indent=4)
        ydl.download([video_url])

    if quiet:
        print(f'\n\nduration {time.time() - t0:.2f}s\n\n')
        # Restore original stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def main():
    download_path = 'demo'

    video_ids = ['0jC-sW5l4_g', 'xspwwNFkxY8', '-l8UyG-3VVc']  #

    for video_id in video_ids:
        # download
        download_youtube_video(download_path, video_id,
                               max_video_length=None, quiet=True)


if __name__ == '__main__':
    main()
