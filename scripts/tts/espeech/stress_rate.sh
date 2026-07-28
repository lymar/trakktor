#!/bin/zsh
# How often does the model stress a word wrong when the text does not say?
#
# The measure leans on Russian vowel reduction: an unstressed «о» is spoken as
# «а», and a recognizer transcribing an unfamiliar word phonetically then writes
# it that way. So "did the target word come back spelled right" stands in for
# "did the stress land right", and `asr gigaam` is the witness.
#
# Two limits, both learned the hard way and worth knowing before trusting a
# number out of this:
#
#   * It only works where a wrong stress changes a vowel the recognizer writes.
#     For a foreign name it may impose its own spelling either way («Голум»,
#     «Беггенс»), and for a homograph both readings are spelled identically —
#     those cases carry no signal at all.
#   * A mark changes the length of the text, and therefore the noise the run
#     starts from, so a single seed conflates the mark with the draw. Hence
#     several seeds per word; one seed produced an apparent regression that did
#     not reproduce.
#
# Usage (needs a converted checkpoint and the gigaam model, both downloaded on
# first use):
#   scripts/tts/espeech/stress_rate.sh tmp/espeech/ref.wav tmp/espeech/ref.txt

set -u
REF_AUDIO=${1:-tmp/espeech/ref.wav}
REF_TEXT_FILE=${2:-tmp/espeech/ref.txt}
SEEDS=(0 1 2)
T=./target/release/trakktor
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# word <TAB> phrase <TAB> the same phrase with the stress marked
read -r -d '' SET <<'SET'
форзаце	На форзаце была наклеена старая карта.	На ф+орзаце была наклеена старая карта.
лоцией	Штурман сверился с лоцией перед выходом в море.	Штурман сверился с л+оцией перед выходом в море.
Мордор	Дорога вела прямо в Мордор, и другой дороги не было.	Дорога вела прямо в М+ордор, и другой дороги не было.
Гондор	Гондор ждал этой помощи много долгих лет.	Г+ондор ждал этой помощи много долгих лет.
Рохана	Всадники Рохана пришли на рассвете третьего дня.	Всадники Р+охана пришли на рассвете третьего дня.
Голлум	Голлум крался за ними по мокрым камням.	Г+оллум крался за ними по мокрым камням.
орки	Орки шли всю ночь и не сделали остановки.	+Орки шли всю ночь и не сделали остановки.
Изенгард	Изенгард стоял в кольце обледенелых скал.	Изенг+ард стоял в кольце обледенелых скал.
Бэггинс	Его звали Бэггинс, и он не любил приключений.	Его звали Б+эггинс, и он не любил приключений.
Мория	Мория оказалась пустой, темной и очень тихой.	М+ория оказалась пустой, темной и очень тихой.
Смеагол	Смеагол еще помнил свое прежнее имя.	Смеаг+ол еще помнил свое прежнее имя.
SET

printf '%-10s %-6s %-7s %-5s %s\n' word seed variant hit "what the recognizer heard"
echo "$SET" | while IFS=$'\t' read -r word plain marked; do
  [ -z "$word" ] && continue
  for seed in $SEEDS; do
    for variant in plain marked; do
      text=$plain
      [ "$variant" = marked ] && text=$marked
      $T tts espeech "$text" --ref-audio "$REF_AUDIO" \
        --ref-text "$(cat "$REF_TEXT_FILE")" --device metal --nfe-step 16 \
        --seed "$seed" -o "$WORK/out.wav" > /dev/null 2>&1
      heard=$($T asr gigaam "$WORK/out.wav" --text 2>/dev/null | tail -1 |
        sed 's/^\[[^]]*\] //')
      hit=MISS
      echo "$heard" | grep -qi -- "$word" && hit=ok
      printf '%-10s %-6s %-7s %-5s %s\n' "$word" "$seed" "$variant" "$hit" "$heard"
    done
  done
done
