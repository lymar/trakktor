async function run() {
  let json_schema = {
    name: "math_reasoning",
    schema: {
      type: "object",
      properties: {
        steps: {
          type: "array",
          items: {
            type: "object",
            properties: {
              explanation: { type: "string" },
              output: { type: "string" },
            },
            required: ["explanation", "output"],
            additionalProperties: false,
          },
        },
        final_answer: { type: "string" },
      },
      required: ["steps", "final_answer"],
      additionalProperties: false,
    },
    strict: true,
  };

  // const pretty = JSON.stringify(json_schema, null, 4);
  // console.log("Using JSON Schema:\n" + pretty);

  try {
    let res = await chat(
      [
        {
          role: "system",
          content:
            "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {
          role: "user",
          content: "how can I solve 8x + 7 = -23",
        },
      ],
      {
        json_schema: json_schema,
      },
    );

    console.log("Response:\n" + res);
  } catch (e) {
    console.error("Error during chat:\n" + e);
  }
}

run();
