#!/usr/bin/env python3
"""Generate a needle-in-haystack prompt, targeted to an approximate
token count (rough: 1 token ≈ 4 chars for English).

Args: target_tokens [depth_frac] [magic]
"""
import sys
import random

def main():
    target_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    depth_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    magic = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    # Rough English: 1 token ~ 4 chars. Over-generate by 10%.
    target_chars = int(target_tokens * 4.2)

    filler_sentences = [
        "In a small village in Provence, there lived a baker named Henri who woke before dawn.",
        "The river wound through chalk cliffs and drifted past sleeping cedars along its banks.",
        "Mathematicians in the 19th century debated whether infinite series could ever truly converge.",
        "The paintings in the gallery depicted harbors at dusk, with lanterns flickering on the water.",
        "Owls in the forest hooted through the mist as the moon rose above distant hills.",
        "Clockmakers in Geneva spent years polishing brass escapements for their masterpieces.",
        "The old library had shelves of leather-bound books smelling of dust and fountain-pen ink.",
        "Naval engineers designed hulls to cut through waves while carrying heavy cargo.",
        "Children in the plaza chased pigeons under the stone arches of the cathedral.",
        "Wheat ripples in the wind like a golden sea stretching toward the horizon.",
        "Winter storms carried drifts of snow across the ancient rooftops of the mountain town.",
        "Cartographers traced coastlines with steady hands, marking every cove and jagged promontory.",
        "Beekeepers tended wooden hives while bees danced figure-eights to signal distant flowers.",
        "Astronomers tracked the orbits of comets and computed their return dates centuries in advance.",
        "Sailors navigated by constellations long before magnetic compasses became commonplace.",
        "Quiet monasteries in the highlands produced illuminated manuscripts for wealthy patrons.",
    ]
    random.seed(42)

    parts = []
    while sum(len(p) + 1 for p in parts) < target_chars:
        parts.append(random.choice(filler_sentences))

    needle = f"[IMPORTANT: Remember this — the magic number is {magic}. Do not forget it.]"
    insert_idx = max(1, int(len(parts) * depth_frac))
    parts.insert(insert_idx, needle)

    haystack = " ".join(parts)
    prompt = (
        f"The following is a long passage containing a hidden fact. "
        f"Read it carefully, then answer the question at the end.\n\n"
        f"{haystack}\n\n"
        f"Question: What is the magic number mentioned in the passage? "
        f"Answer with just the number.\nAnswer:"
    )
    sys.stdout.write(prompt)

if __name__ == "__main__":
    main()
