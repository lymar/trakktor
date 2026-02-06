export async function run() {
    try {
        let audio_file = await readFile("audio.mp3");
        await preview(audio_file, LONG_TEXT_EN);
        // await preview(audio_file);
    } catch (e) {
        console.error("Error:\n" + e);
    }
}

const LONG_TEXT_EN = `At dawn, the valley of Greyfen looked harmless—mist pooled between the reeds, and the river carried pale leaves as if it had never learned the weight of winter. But the old stones on the hillside remembered other mornings: mornings when the sky rang like struck iron and the crows flew in perfect, silent lines. Liora tightened the strap of her satchel and counted her breaths—one, two, three—because that was what her mother had taught her to do when stories tried to step out of the dark and into the world. Ahead, the road bent toward the ruined watchtower, where a single lantern still burned in the broken window, steady and waiting, as though someone inside had been awake for years.`;
