# CS256

Steps to Deploy this project:

1.> Fork this repo

2.> Sign up for a Streamlit account and deploy this app, following the steps as mentioned in https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app or find detailed instructions at the end of this document. 


3.> Visit your newly created app using the URL you generated in Step 2!

### Challenges Faced:
1.> Streamlit Community Cloud has a resource constraint, and TFHub, T5 Models are 1GB each. This may cause the deployed app to crash sometimes, which may require a rebooting of the app. 

### Streamlit App Deployment Link for Reference:
Link 1: 
https://t5s8c9gjpppa9mpqf5kta9.streamlit.app/

Link 2:
https://team3needextracreditplease.streamlit.app/


### Detailed Steps for Deploying this project: 


Deploying an app on Streamlit Community Cloud involves a few steps. Please note that the details and steps may evolve over time, so it's always a good idea to check the Streamlit documentation for the latest information. This is current as of Dec 14 2023.

##### Prepare Your Streamlit App:

Make sure your Streamlit app is ready and functional on your local machine.
Ensure that your app dependencies are listed in a requirements.txt file or similar.

##### Create a GitHub Repository:

Upload your Streamlit app code to a GitHub repository. If you don't have a GitHub account, you'll need to create one.

##### Add a requirements.txt File:

Include a requirements.txt file in your GitHub repository to specify the dependencies needed for your app.

##### Configure Your Streamlit App:

Ensure your Streamlit app is configured to use the correct port. You might need to set the port explicitly using the --server.port option in your Streamlit command.

##### Sign in to Streamlit Sharing:

Visit the Streamlit Sharing platform.
Sign in using your GitHub account.

##### Create a New App:

Click on the "Create a new app" button on the Streamlit Sharing dashboard.
Follow the instructions to connect your GitHub repository.

##### Specify the Repository and Branch:

Choose the repository and branch that contains your Streamlit app code.

##### Configure the App Settings:

Set the desired configuration options for your app (e.g., number of instances, environment variables).

##### Deploy Your App:

Click on the "Deploy" button to start the deployment process.

##### Wait for Deployment:

Streamlit Sharing will start building and deploying your app. You can monitor the progress on the dashboard.

##### Access Your Deployed App:

Once the deployment is successful, you'll be provided with a link to access your live Streamlit app.

##### Share Your App:

Share the provided link with others to let them access your deployed Streamlit app.
Remember that Streamlit Sharing has certain limitations on resource usage, and it's designed for sharing lightweight applications and data apps. If you have specific requirements or need more resources, you might consider other hosting options.
