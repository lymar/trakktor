async function run() {
  let wikiText = await httpGetText("https://en.wikipedia.org/wiki/Main_Page");
  console.log("Fetched text:" + wikiText);

  let markdown = htmlToMarkdown(wikiText);
  console.log("Converted to Markdown:\n" + markdown);

  try {
    let res = await chat(
      [
        { role: "system", content: prompt },
        { role: "user", content: markdown },
      ],
      { json_schema: json_schema },
    );

    console.log("Response:\n" + res);
  } catch (e) {
    console.error("Error during chat:\n" + e);
  }
}

run();

let json_schema = {
  name: "wikipedia_on_this_day",
  schema: {
    type: "object",
    properties: {
      events: {
        type: "array",
        description:
          'Extracted "On this day" entries as (year, description) pairs.',
        items: {
          type: "object",
          properties: {
            year: {
              type: "string",
              description:
                'The year label exactly as shown (e.g., "1066", "44 BC").',
            },
            description: {
              type: "string",
              description:
                "Plain text description of the event with all links/URLs/Markdown removed.",
            },
          },
          required: ["year", "description"],
          additionalProperties: false,
        },
      },
    },
    required: ["events"],
    additionalProperties: false,
  },
  strict: true,
};

let prompt = `You are given the Markdown of the first page of Wikipedia for a specific date. Your task is to extract the entries from the section titled exactly **"On this day"** and return them as JSON that matches the provided JSON Schema.

Rules:

* Find the **"On this day"** section and extract the list of events in that section.
* Each event must become one object with:

  * 'year': the year shown for the event (keep it exactly as shown, as text; e.g., "44 BC" is allowed).
  * 'description': the event text as plain readable text.
* Remove all links and link markup from the description:

  * Convert Markdown links like '[Text](URL)' to 'Text'.
  * Remove reference markers/footnotes like '[...]', '^', citation templates, or inline citation numbers if present.
* Do not include URLs, brackets, or Markdown formatting in 'description'.
* Do not include anything outside the **"On this day"** section (ignore “Events”, “Births”, “Deaths”, “Holidays”, etc., unless they are explicitly inside “On this day” on that page).
* Preserve the original order of events as they appear in the section.
* Output **only** valid JSON that conforms to the schema.

Input: the Markdown content of the Wikipedia page.`;
