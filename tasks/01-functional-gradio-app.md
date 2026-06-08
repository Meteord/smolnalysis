# Functional Gradio App (MVP + extensions)

- Parent list: [task_list.md](task_list.md)
- Related vision: [vision.md](vision.md)

## Status

- Progress: MVP implemented in `app/` using Gradio Server Mode and a fullscreen OpenUI chat frontend
- Owner:
- Target date:

## Checklist

- [x] Build MVP Gradio interface for CSV upload
- [x] Implement baseline data analysis flow
- [x] Add extension support for OpenUI commands
- [x] Add chatbot-style OpenUI React rendering flow ([01.1-openui-support-in-gradio-app.md](01.1-openui-support-in-gradio-app.md))
- [x] Add at least one demo dataset for validation

## Notes

- App entry point: `app/app.py`
- Local dependencies: `app/requirements.txt`
- Demo dataset: `app/examples/demo_cities.csv`
- Current frontend: OpenUI `FullScreen` chat served by `gr.Server`
