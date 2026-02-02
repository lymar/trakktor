async function run() {
    try {
        console.log("yo!");
        let audio_file = await readFile("audio.mp3");
    } catch (e) {
        console.error("Error:\n" + e);
    }
}

run();
