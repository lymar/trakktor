# Where should we write the GIF?
Output tts.gif
Output tts.mp4

# Set up a 1200x600 terminal with 46px font.
Set FontSize 22
Set Width 1280
Set Height 800
Set TypingSpeed 30ms

Hide
Type "cd ../../"
Enter
Type "clear"
Enter

Show

# Type "head -c 214 examples/text_to_speech.js"
Type "bat examples/text_to_speech.js"
Sleep 500ms
Enter

Sleep 7s
Type "q"

Type "cargo run --bin dev_run_js -- examples/text_to_speech.js"
Sleep 500ms
Enter

Hide

Wait@30s /Scroll/
Show

Sleep 2s

Down@500ms 2

Sleep 2s

Up@500ms 2

Sleep 9s

Type "x"
Sleep 2000ms
