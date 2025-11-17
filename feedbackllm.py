import os, requests, json
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_ENDPOINT = os.getenv('GROQ_ENDPOINT')

class CoachLLM:
    def __init__(self, groq_endpoint=None, groq_key=None, model_name='gpt-like'):
        self.groq_endpoint = groq_endpoint or GROQ_ENDPOINT
        self.groq_key = groq_key or GROQ_API_KEY
        self.model = model_name

    def _build_prompt(self, sport, level, issues, features, ref_context=None):
        prompt = {
            'system':'You are a professional basketball shooting coach. Provide concise, actionable drills and a one-sentence summary.',
            'sport': sport,
            'athlete_level': level,
            'observed_issues': issues,
            'numeric_summary': {k: round(float(sum(v)/len(v)),2) if isinstance(v,list) and len(v)>0 else v for k,v in features.items() if k!='fps'},
            'reference_context': ref_context
        }
        return json.dumps(prompt, indent=2)

    def generate_tip(self, sport, level, issues, features, ref_context=None):
        prompt = self._build_prompt(sport, level, issues, features, ref_context)
        fall = {
            'summary':'Focus on elbow extension timing and hip drive.',
            'drills':['1) Slow-motion elbow extension reps','2) Hip-drive timed jumps (metronome)','3) Video mirror feedback 3x/week'],
            'motivation':'Consistency beats intensity — small daily wins.'
        }
        if self.groq_endpoint and self.groq_key:
            try:
                headers = {'Authorization':f'Bearer {self.groq_key}', 'Content-Type':'application/json'}
                payload = {'model':self.model, 'prompt':prompt, 'max_tokens':300}
                r = requests.post(self.groq_endpoint, json=payload, headers=headers, timeout=12)
                if r.status_code==200:
                    return {'text': r.json().get('text', r.text), 'prompt': prompt}
            except Exception as e:
                print('Groq call failed:', e)
        out = fall['summary'] + '\n' + '\n'.join(fall['drills']) + '\n' + fall['motivation']
        return {'text': out, 'prompt': prompt}
