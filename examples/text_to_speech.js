async function run() {
    try {
        let text = LONG_TEXT_EN;
        let audio = await textToSpeech(text, { speed: 1.0, voice: "marin" });
        await preview(audio, text);
    } catch (e) {
        console.error("Error:\n" + e);
    }
}

const LONG_TEXT_EN = `At dawn, the valley of Greyfen looked harmless—mist pooled between the reeds, and the river carried pale leaves as if it had never learned the weight of winter. But the old stones on the hillside remembered other mornings: mornings when the sky rang like struck iron and the crows flew in perfect, silent lines. Liora tightened the strap of her satchel and counted her breaths—one, two, three—because that was what her mother had taught her to do when stories tried to step out of the dark and into the world. Ahead, the road bent toward the ruined watchtower, where a single lantern still burned in the broken window, steady and waiting, as though someone inside had been awake for years.

She reached the tower as the first sunlight thinned the mist, and the lantern’s glow did not fade—it swallowed the dawn instead, turning the cracked stones warm as skin. No footsteps answered her call. Only a faint, rhythmic scrape rose from somewhere above, like a quill writing on dry parchment. Liora stepped through the doorframe and felt the air change, heavy with the smell of rain that hadn’t fallen yet. On the inner wall, a ring of chalk symbols circled the room, each mark sharp and fresh, and in the center lay a small copper bowl filled with water so still it reflected not her face, but a sky crowded with unfamiliar stars. When she leaned closer, one star winked—slowly, deliberately—as if it had been waiting for her name.`;

const LONG_TEXT_RU = `На рассвете долина Грейфен казалась безобидной: туман собирался в лужицы меж камышей, а река несла бледные листья, словно никогда не знала тяжести зимы. Но старые камни на склоне помнили другие утра — утра, когда небо звенело, как ударенное железо, и вороны летели ровными, безмолвными линиями. Лиора крепче затянула ремень своей сумки и сосчитала вдохи — раз, два, три, — потому что мать учила её делать так, когда истории пытаются выйти из темноты и вступить в мир. Впереди дорога изгибалась к разрушенной сторожевой башне, где в разбитом окне всё ещё горел один-единственный фонарь — ровно и терпеливо, будто кто-то внутри не спал уже много лет.

Она подошла к башне, когда первый солнечный свет начал разрежать туман, — и сияние фонаря не поблекло: напротив, оно как будто поглотило рассвет, делая потрескавшиеся камни тёплыми, почти живыми. На её зов никто не ответил — лишь где-то наверху доносилось тихое, размеренное поскребывание, словно перо выводит слова по сухому пергаменту. Лиора переступила через дверной проём и почувствовала, как меняется воздух: он стал густым, тяжёлым, пахнущим дождём, который ещё не пролился. На внутренней стене мелом был вычерчен круг символов — каждый знак острый и свежий, — а в центре стояла маленькая медная чаша с водой такой неподвижной, что она отражала не её лицо, а небо, тесно заполненное незнакомыми звёздами. Лиора наклонилась ближе — и одна звезда моргнула, медленно и нарочно, будто давно ждала, когда прозвучит её имя.`;

run();
