import csv, random, os
random.seed(42)
TOXIC = [
    "You are such an idiot, nobody cares about your stupid opinion.",
    "Go kill yourself, worthless piece of garbage.",
    "This is the dumbest thing I have ever read, you moron.",
    "You people are all trash, get out of this country.",
    "I hate you so much, you should disappear forever.",
    "What a disgusting human being you are.",
    "You are pathetic and nobody likes you.",
    "Shut up you absolute clown, your ideas are garbage.",
    "This is absolute crap written by a brainless fool.",
    "You are the worst person alive, truly despicable.",
    "Die already you horrible excuse for a human.",
    "Stop talking, you are so incredibly stupid it hurts.",
    "Everyone hates you and your worthless comments.",
    "Your kind is ruining everything, get lost.",
    "I would like to punch you right in your stupid face.",
    "You filthy animal, you do not deserve to breathe.",
    "What a piece of garbage you are, go away forever.",
    "Nobody wants your opinion, just shut your mouth.",
    "You are utterly useless and a waste of space.",
    "I despise people like you with every fiber of my being.",
    "Toxic garbage comment by a trash human.",
    "You disgusting rat, crawl back under your rock.",
    "Absolute moron alert, this person is brain dead.",
    "Stop polluting the internet with your idiotic takes.",
    "You are the reason this world is going downhill fool.",
]
NON_TOXIC = [
    "I really enjoyed reading this article, very informative.",
    "Thanks for sharing your perspective on this matter.",
    "I disagree with some points but appreciate the discussion.",
    "This is a great community, everyone is so helpful.",
    "Could you please elaborate on that point a bit more?",
    "I think there are valid arguments on both sides of this issue.",
    "Wonderful post, looking forward to reading more from you.",
    "I learned something new today, thank you for sharing.",
    "This has been a very productive conversation.",
    "Happy to help if you have any more questions.",
    "I see where you are coming from, interesting viewpoint.",
    "The data presented here is quite compelling.",
    "Let us try to keep the discussion civil and respectful.",
    "What a fascinating topic, I had never considered that before.",
    "Great points all around, very well argued.",
    "I appreciate you taking the time to explain this so clearly.",
    "Looking forward to the next installment of this series.",
    "This is exactly the kind of content I come here for.",
    "You explained that beautifully, thank you.",
    "I think we can all agree that more research is needed.",
    "Really thoughtful analysis, I will share this with my team.",
    "Thank you for your patience in explaining this.",
    "That is a nuanced perspective I had not considered.",
    "I enjoyed the discussion, everyone made good points.",
    "Keep up the excellent work, this community is fantastic.",
    "I found this very useful for my project, appreciated.",
    "Respectfully disagree, but I understand your reasoning.",
    "This was a pleasure to read, very well written.",
    "I will definitely try this approach, thanks for the tip.",
    "Excellent resource, bookmarking this for later.",
]
rows = [{"comment_text": c, "toxic": 1} for c in TOXIC] + [{"comment_text": c, "toxic": 0} for c in NON_TOXIC]
expanded = []
suffixes = ["Really.", "Seriously.", "Think about it.", "No doubt.", "Period."]
for _ in range(10):
    for row in rows:
        t = row["comment_text"]
        if random.random() > 0.6:
            t = t + " " + random.choice(suffixes)
        expanded.append({"comment_text": t, "toxic": row["toxic"]})
random.shuffle(expanded)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["comment_text","toxic"])
    w.writeheader(); w.writerows(expanded)
print(f"Generated {len(expanded)} rows -> {out}")
