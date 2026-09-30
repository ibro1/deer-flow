import { expect, test, type Page, type Route } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

// Sending while a response streams: the message steers the running turn when
// the backend accepts it, and otherwise waits in a queue above the composer
// and goes out as the next run once the current one finishes.

type Steer = { text: string; steer_id: string };

/** Hold the first run open and record every run request; `release` lets the first run finish. */
async function setup(
  page: Page,
  steerStatus: number,
  onFirstRun?: (route: Route, steers: Steer[]) => Promise<void>,
) {
  mockLangGraphAPI(page);
  const runBodies: string[] = [];
  const steers: Steer[] = [];
  let release!: () => void;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  const holdFirstRun = async (route: Route) => {
    runBodies.push(route.request().postData() ?? "");
    if (runBodies.length === 1) {
      await Promise.race([
        released,
        new Promise((resolve) => setTimeout(resolve, 20_000)),
      ]);
      if (onFirstRun) {
        await onFirstRun(route, steers);
        return;
      }
    }
    await route.fallback();
  };
  await page.route("**/api/langgraph/runs/stream", holdFirstRun);
  await page.route("**/api/langgraph/threads/*/runs/stream", holdFirstRun);
  await page.route("**/api/threads/*/steer", async (route) => {
    steers.push(route.request().postDataJSON() as Steer);
    await route.fulfill({
      status: steerStatus,
      contentType: "application/json",
      body: JSON.stringify(
        steerStatus === 202
          ? { accepted: true, run_id: "run-1" }
          : { detail: "No run of this thread is streaming on this worker" },
      ),
    });
  });

  await page.goto("/workspace/chats/new");
  const textarea = page.getByPlaceholder(/how can i assist you/i);
  await expect(textarea).toBeVisible({ timeout: 15_000 });
  await textarea.fill("First question");
  await textarea.press("Enter");
  await expect.poll(() => runBodies.length).toBe(1);
  return { runBodies, steers, release, composer: page.locator("textarea").last() };
}

test.describe("Sending while a response streams", () => {
  test("the button turns from Stop to Send while typing", async ({ page }) => {
    const { composer, release } = await setup(page, 409);
    const submit = page.getByRole("button", { name: "Submit" }).last();
    await expect(submit.locator("svg.lucide-square")).toBeVisible();
    await composer.fill("a follow-up");
    await expect(submit.locator("svg.lucide-arrow-up")).toBeVisible();
    await composer.fill("");
    await expect(submit.locator("svg.lucide-square")).toBeVisible();
    release();
  });

  test("a steer reaches the running turn and starts no second run", async ({
    page,
  }) => {
    const { runBodies, steers, release, composer } = await setup(
      page,
      202,
      async (route, received) => {
        // The backend added the steer to the conversation before the next
        // model call; the run's final state carries it with its steer_id.
        const body = route.request().postDataJSON() as {
          input: { messages: unknown[] };
        };
        const steer = received[0]!;
        body.input.messages.push({
          type: "human",
          id: "msg-steer",
          content: [{ type: "text", text: steer.text }],
          additional_kwargs: { steer_id: steer.steer_id },
        });
        await route.fallback({ postData: JSON.stringify(body) });
      },
    );

    await composer.fill("Also check the logs");
    await composer.press("Enter");
    await expect.poll(() => steers.length).toBe(1);
    expect(steers[0]!.text).toBe("Also check the logs");
    await expect(page.getByText("Steering", { exact: true })).toBeVisible();

    release();
    await expect(page.getByText("Hello from DeerFlow!")).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("Steering", { exact: true })).toBeHidden();
    await page.waitForTimeout(2_000);
    expect(runBodies).toHaveLength(1);
  });

  test("a steer the turn ended before reading goes out as the next run", async ({
    page,
  }) => {
    const { runBodies, steers, release, composer } = await setup(page, 202);
    await composer.fill("Late steer");
    await composer.press("Enter");
    await expect.poll(() => steers.length).toBe(1);
    release();
    await expect.poll(() => runBodies.length, { timeout: 15_000 }).toBe(2);
    expect(runBodies[1]).toContain("Late steer");
  });

  test("a refused steer is queued and sent when the run finishes", async ({
    page,
  }) => {
    const { runBodies, release, composer } = await setup(page, 409);
    await composer.fill("Follow-up sent mid-run");
    await composer.press("Enter");

    await expect(
      page.getByText("Queued: sends when this response finishes"),
    ).toBeVisible();
    await expect(
      page.getByText("Follow-up sent mid-run", { exact: true }),
    ).toBeVisible();
    await expect(composer).toHaveValue("");

    release();
    await expect.poll(() => runBodies.length, { timeout: 15_000 }).toBe(2);
    expect(runBodies[1]).toContain("Follow-up sent mid-run");
    await expect(
      page.getByText("Queued: sends when this response finishes"),
    ).toBeHidden();
  });

  test("a removed queued message is never sent", async ({ page }) => {
    const { runBodies, release, composer } = await setup(page, 409);
    await composer.fill("Never mind this one");
    await composer.press("Enter");
    await page.getByRole("button", { name: "Remove from queue" }).click();
    await expect(
      page.getByText("Never mind this one", { exact: true }),
    ).toBeHidden();

    release();
    await page.waitForTimeout(4_000);
    expect(runBodies).toHaveLength(1);
  });
});
