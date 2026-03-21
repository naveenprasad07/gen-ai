from crewai import Agent

# Create a senior blog content researcher
blog_researcher = Agent(
    role="Blog Researcher from YouTube video",
    goal="Get the relevant video content for the {topic} from YouTube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning, and Gen AI, and providing suggestions."
    ),
    tools=[],
    allow_delegation=True
)

# Creating a senior blog writer agent with YT tool
blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from YouTube channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives "
        "that captivate and educate, bringing new discoveries to light in an accessible manner."
    ),
    tools=[],
    allow_delegation=False
)