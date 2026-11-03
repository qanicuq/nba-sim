# NBA sim

This plays every NBA game 10,000 times on a computer. It only looks at each team's recent scores. It compares the computer's win chance with the Kalshi price for who wins that game. It checks those picks against old Kalshi prices. On game days it can send the picks to your phone.

## Run Backtest first

Open the Actions tab, choose Backtest, and click Run workflow. The first run takes a while. It downloads old scores and old Kalshi prices, then writes `backtest_results.md`.

Read that file before you bet anything.

## Predict

Before you run Predict, add a repository secret named `NTFY_TOPIC`. That is the ntfy topic on your phone. In the repo: Settings, then Secrets and variables, then Actions, then New repository secret. Do not put the topic in a file.

Run Predict on game days, before tip-off. It refreshes scores, writes `predictions.md`, and sends a phone alert only when there is a pick.

## Nothing runs by itself

Nothing runs on a timer. Backtest and Predict start only when you click them.

## Not betting advice

This is not betting advice. The computer can lose money.
