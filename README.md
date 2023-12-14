# CS256

Steps to Deploy this 

1.> Fork this repo
2.> Sign up for a Streamlit account and deploy this app
3.> Profit

### Known issues:
1.> Feature Vector model is not a "model" - so its not pickleable and not compatible per se with this app... needs fixes 
2.> TFhub and T5 models are too big to pull, and too big to pickle - need workaround for this 
3.> The UI to pick the models currently is hardcoded to pick word2vec, needs to be generalized to the 4 models once done