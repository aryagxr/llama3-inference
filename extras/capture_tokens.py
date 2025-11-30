#!/usr/bin/env python3
"""
Script to capture token generation and create a clean GIF showing tokens appearing progressively.
Usage: python capture_tokens.py <file_to_run> [--max-tokens N] [--fps FPS]
"""

import subprocess
import sys
import time
import threading
import re
from PIL import Image, ImageDraw, ImageFont
import argparse
import os

# Try to import imageio for GIF creation, fallback to PIL
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

class TokenCapture:
    def __init__(self, fps=10):
        self.fps = fps
        self.tokens = []  # List of (timestamp, token_text) tuples
        self.prompt = None
        self.start_time = None
        self.lock = threading.Lock()
        self.running = True
        self.token_pattern = re.compile(r"Step \d+: Generated token '([^']+)'")
        
    def start(self):
        self.start_time = time.time()
        
    def add_line(self, line):
        """Parse line and extract token if it's a token generation line"""
        if self.start_time is None:
            self.start_time = time.time()
        
        timestamp = time.time() - self.start_time
        
        # Try to extract token from "Step X: Generated token 'Y'"
        match = self.token_pattern.search(line)
        if match:
            token_text = match.group(1)
            with self.lock:
                self.tokens.append((timestamp, token_text))
        
        # Try to extract prompt from initial output
        if self.prompt is None and ("Starting generation" in line or "Initial prompt" in line):
            # Look for prompt in the next few lines or extract from tokenizer output
            pass
    
    def get_tokens_at_time(self, t):
        """Get all tokens that should be visible at time t"""
        with self.lock:
            visible_tokens = [token for timestamp, token in self.tokens if timestamp <= t]
            return visible_tokens
    
    def get_all_data(self):
        with self.lock:
            return self.tokens.copy()

def read_output(process, capture):
    """Read output from process in real-time"""
    try:
        for line in iter(process.stdout.readline, b''):
            if not capture.running:
                break
            line_str = line.decode('utf-8', errors='replace').rstrip()
            if line_str:
                capture.add_line(line_str)
                # Don't print to console for cleaner output
    except Exception as e:
        print(f"Error reading output: {e}", file=sys.stderr)
    finally:
        capture.running = False

def find_font(size=14):
    """Try to find a monospace font"""
    font_paths = [
        '/System/Library/Fonts/Menlo.ttc',  # macOS
        '/System/Library/Fonts/SF-Mono-Regular.otf',  # macOS
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',  # Linux
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',  # Linux
        'C:/Windows/Fonts/consola.ttf',  # Windows
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    
    # Fallback to default font
    try:
        return ImageFont.load_default()
    except:
        return None

def create_token_frame(prompt, generated_tokens, width=1200, height=600, font_size=20, padding=40, title_text=None):
    """Create a frame showing prompt + progressively generated tokens"""
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    font = find_font(font_size)
    if font is None:
        font = ImageFont.load_default()
    
    title_font = find_font(font_size + 6)
    if title_font is None:
        title_font = font
    
    y = padding
    
    # Draw title if provided
    if title_text:
        draw.text((padding, y), title_text, fill='#00ff00', font=title_font)
        y += int(font_size * 2.5)
    
    # Draw prompt label and text (in a lighter color)
    if prompt:
        draw.text((padding, y), "User:", fill='#888888', font=font)
        y += int(font_size * 1.2)
        
        # Word wrap the prompt
        prompt_lines = wrap_text(prompt, width - 2 * padding, font, draw)
        for line in prompt_lines:
            if y + font_size + 10 > height - padding - 50:  # Leave room for generated text
                break
            draw.text((padding + 20, y), line, fill='#aaaaaa', font=font)
            y += font_size + 6
        y += int(font_size * 0.8)
    
    # Draw assistant label
    draw.text((padding, y), "Assistant:", fill='#00ff88', font=font)
    y += int(font_size * 1.2)
    
    # Draw generated tokens (in bright color)
    generated_text = ''.join(generated_tokens)
    if generated_text:
        # Word wrap the generated text
        gen_lines = wrap_text(generated_text, width - 2 * padding, font, draw)
        for line in gen_lines:
            if y + font_size + 10 > height - padding:
                break
            draw.text((padding + 20, y), line, fill='#00ff88', font=font)
            y += font_size + 6
    
    # Draw cursor (blinking block cursor)
    cursor_x = padding + 20
    if generated_text:
        # Calculate where cursor should be based on last line
        if gen_lines:
            last_line = gen_lines[-1]
            bbox = draw.textbbox((0, 0), last_line, font=font)
            cursor_x = padding + 20 + (bbox[2] - bbox[0])
            cursor_y = y - font_size - 6
        else:
            cursor_y = y
    else:
        cursor_y = y
    
    draw.text((cursor_x, cursor_y), '▊', fill='#00ff88', font=font)
    
    return img

def wrap_text(text, max_width, font, draw):
    """Wrap text to fit within max_width"""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        # Test if adding this word would exceed width
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [text]

def extract_prompt_from_file(file_path):
    """Try to extract the user prompt from the Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Look for prompt = "..."
            match = re.search(r'prompt\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                full_prompt = match.group(1)
                # Extract just the user message part (between user tags)
                # Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|><br><br>USER_MESSAGE<|eot_id|>...
                user_match = re.search(r'<\|start_header_id\|>user<\|end_header_id\|><br><br>([^<]+)<\|eot_id\|>', full_prompt)
                if user_match:
                    return user_match.group(1).strip()
                # Fallback: return a cleaned version
                return full_prompt.replace('<|begin_of_text|>', '').replace('<|start_header_id|>user<|end_header_id|><br><br>', 'User: ').replace('<|eot_id|>', '').replace('<|start_header_id|>assistant<|end_header_id|><br><br>', '\n\nAssistant: ')
    except:
        pass
    return None

def create_gif_with_imageio(frames, output_path, fps=10):
    """Create GIF using imageio (better quality)"""
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

def create_gif_with_pil(frames, output_path, fps=10):
    """Create GIF using PIL (fallback)"""
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

def main():
    parser = argparse.ArgumentParser(description='Capture token generation and create clean token GIF')
    parser.add_argument('file', help='Python file to run (e.g., 01-naive.py)')
    parser.add_argument('--max-tokens', type=int, default=50, 
                       help='Maximum number of tokens to generate (default: 50)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIF (default: 10)')
    parser.add_argument('--width', type=int, default=1200,
                       help='GIF width in pixels (default: 1200)')
    parser.add_argument('--height', type=int, default=600,
                       help='GIF height in pixels (default: 600)')
    parser.add_argument('--font-size', type=int, default=20,
                       help='Font size (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output GIF filename (default: <file>_tokens.gif)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt text (default: extracted from file)')
    
    args = parser.parse_args()
    
    # Resolve file path - try relative to current dir, then relative to script dir
    if not os.path.exists(args.file):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, '..', args.file)
        if os.path.exists(potential_path):
            args.file = os.path.abspath(potential_path)
        elif not os.path.isabs(args.file):
            # Try from current working directory
            cwd_path = os.path.join(os.getcwd(), args.file)
            if os.path.exists(cwd_path):
                args.file = os.path.abspath(cwd_path)
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    
    # Determine output filename
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        args.output = f"{base_name}_tokens.gif"
    
    # Extract prompt
    prompt = args.prompt
    if prompt is None:
        prompt = extract_prompt_from_file(args.file)
    if prompt is None:
        prompt = "Prompt: [extracted from model output]"
    
    print(f"Running {args.file} and capturing tokens...")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Output will be saved to {args.output}")
    print("-" * 60)
    
    # Create capture object
    capture = TokenCapture(fps=args.fps)
    capture.prompt = prompt
    capture.start()
    
    # Set up environment for unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if args.max_tokens:
        env['MAX_NEW_TOKENS'] = str(args.max_tokens)
    
    # Start the process
    process = subprocess.Popen(
        [sys.executable, args.file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.abspath(args.file)) or '.'
    )
    
    # Start reading output in a separate thread
    reader_thread = threading.Thread(target=read_output, args=(process, capture))
    reader_thread.daemon = True
    reader_thread.start()
    
    # Wait for process to complete
    process.wait()
    capture.running = False
    reader_thread.join(timeout=5)
    
    # Get all captured tokens
    tokens_data = capture.get_all_data()
    
    if not tokens_data:
        print("No tokens captured! Make sure the script outputs 'Step X: Generated token Y' format.")
        sys.exit(1)
    
    print(f"\nCaptured {len(tokens_data)} tokens")
    print("Creating GIF frames...")
    
    # Determine time range
    if tokens_data:
        max_time = tokens_data[-1][0]
        # Add some padding at the end to show final result
        total_duration = max_time + 2.0
        total_frames = int(total_duration * args.fps) + 1
    else:
        total_frames = 1
        max_time = 1.0
    
    # Get base filename for title
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    title = f"{base_name} - Token Generation"
    
    # Create frames
    frames = []
    for frame_idx in range(total_frames):
        t = frame_idx / args.fps
        visible_tokens = capture.get_tokens_at_time(t)
        # visible_tokens is already a list of token strings, no need to unpack
        frame = create_token_frame(prompt, visible_tokens, args.width, args.height, args.font_size, title_text=title)
        frames.append(frame)
        
        if frame_idx % 10 == 0:
            print(f"Created {frame_idx + 1}/{total_frames} frames...", end='\r')
    
    print(f"\nCreated {len(frames)} frames")
    print("Saving GIF...")
    
    # Save GIF
    try:
        if HAS_IMAGEIO:
            create_gif_with_imageio(frames, args.output, args.fps)
        else:
            print("Note: imageio not available. Using PIL (slower). Install with 'pip install imageio' for better performance.")
            create_gif_with_pil(frames, args.output, args.fps)
    except Exception as e:
        print(f"Error creating GIF: {e}")
        print("Trying alternative method...")
        try:
            create_gif_with_pil(frames, args.output, args.fps)
        except Exception as e2:
            print(f"Failed to create GIF: {e2}")
            sys.exit(1)
    
    print(f"✓ GIF saved to {args.output}")
    if tokens_data:
        print(f"  Duration: {tokens_data[-1][0]:.2f} seconds")
        print(f"  Tokens generated: {len(tokens_data)}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {args.fps}")

if __name__ == '__main__':
    main()

