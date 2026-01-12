# FluxFish Deployment Guide

This guide will help you deploy FluxFish on Lichess.

## Prerequisites

- Python 3.8+
- Your trained `fluxfish.nnue` model
- A Lichess account

## Step 1: Upgrade Your Lichess Account to a Bot Account

⚠️ **WARNING**: This is **IRREVERSIBLE**! Your account will become a bot account and cannot play as a human anymore.

1. Go to https://lichess.org and log in (or create a new account specifically for the bot)
2. Go to https://lichess.org/account/oauth/token/create
3. Create a new API token with these scopes:
   - `bot:play` - Play as a bot
   - `challenge:read` - Read incoming challenges
   - `challenge:write` - Create/accept/decline challenges
4. Copy the token (you'll only see it once!)

5. Upgrade to bot account by running:
```bash
curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Step 2: Install lichess-bot

```bash
# Clone lichess-bot
cd /root
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

## Step 3: Configure

1. Copy our config to lichess-bot:
```bash
cp /root/fluxfish-nnue/config.yml /root/lichess-bot/config.yml
```

2. Edit the config and add your token:
```bash
nano /root/lichess-bot/config.yml
# Replace YOUR_LICHESS_BOT_TOKEN_HERE with your actual token
```

## Step 4: Test the UCI Interface

Before connecting to Lichess, test that the UCI interface works:

```bash
cd /root/fluxfish-nnue
python3 uci.py
```

Then type these commands:
```
uci
isready
position startpos moves e2e4
go movetime 2000
quit
```

You should see the engine respond with a best move.

## Step 5: Run the Bot

```bash
cd /root/lichess-bot
python3 lichess-bot.py
```

The bot will now:
- Connect to Lichess
- Accept incoming challenges
- Play games automatically!

## Step 6: Keep it Running (Optional)

To keep the bot running in the background:

```bash
# Using screen
screen -S fluxfish
cd /root/lichess-bot
python3 lichess-bot.py
# Press Ctrl+A, then D to detach

# To reattach later:
screen -r fluxfish
```

Or using nohup:
```bash
cd /root/lichess-bot
nohup python3 lichess-bot.py &
```

## Troubleshooting

### "Engine not responding"
- Make sure `uci.py` is executable: `chmod +x /root/fluxfish-nnue/uci.py`
- Check that `fluxfish.nnue` exists in the fluxfish-nnue directory

### "Invalid token"
- Make sure you copied the full token
- Check that the token has the required scopes

### "Account is not a bot account"
- Run the upgrade curl command from Step 1

## Monitoring Your Bot

- View your bot's profile: `https://lichess.org/@/YOUR_BOT_USERNAME`
- Watch games live on the Lichess website
- Check logs in the terminal

## Challenge Your Bot

Once running, you can challenge your bot:
1. Go to your bot's profile on Lichess
2. Click "Challenge to a game"
3. Select time control and play!

Others can also find and challenge your bot on Lichess.

---

Good luck with FluxFish! 🐟♟️
