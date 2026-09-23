from kaggle_environments import make
import json, webbrowser


if __name__ == "__main__":
    with open("replay.json", "r") as f:
        episode_data = json.load(f)
        
    env = make("kaggriculture", steps=episode_data["steps"], debug=True)

    html_output = env.render(mode="html", width=1000, height=750)
    with open("render.html", "w", encoding="utf-8") as f:
        f.write(html_output)
    webbrowser.open("render.html")