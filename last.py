
# the final version of commercial - employee based thingy


import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time 

# Page configuration
st.set_page_config(page_title="AI Environmental Impact Management Framework", layout="wide")

# Helper functions for mock data
def generate_mock_emissions_data():
    dates = pd.date_range(start='2024-01-01', end='2024-11-24', freq='D')
    emissions = np.random.normal(100, 20, len(dates))
    return pd.DataFrame({'Date': dates, 'Emissions (kg CO₂e)': emissions})

def generate_mock_project_data():
    return pd.DataFrame({
        'Project': ['Model A', 'Model B', 'Model C', 'Model D'],
        'Emissions': np.random.normal(100, 20, 4),
        'Energy': np.random.normal(500, 50, 4),
        'Water': np.random.normal(200, 30, 4),
        'Efficiency': np.random.uniform(0.7, 0.95, 4)
    })

def generate_mock_team_data():
    return pd.DataFrame({
        'Member': ['Alice', 'Bob', 'Charlie', 'David'],
        'Projects': [3, 4, 2, 5],
        'Impact Score': np.random.uniform(70, 95, 4),
        'Badges': [4, 3, 2, 5]
    })

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login system
def login_page():
    st.title("Welcome to AI Environmental Impact Management Framework")
    
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")
    with col2:
        user_type = st.selectbox("User Type", ["Employee", "Customer"])
    
    if st.button("Login"):
        st.session_state.user_type = user_type.lower()
        st.session_state.logged_in = True
        st.experimental_user()

# Sidebar navigation
def sidebar_nav():
    st.sidebar.title("AIMF Navigation")
    
    if st.session_state.user_type == "employee":
        return st.sidebar.radio(
            "Select a Section:",
            ["Project Dashboard",
             "Development Tools",
             "Team Collaboration",
             "Certifications & Rewards",
             "Impact Simulator",
             "Training Hub"]
        )
    else:
        return st.sidebar.radio(
            "Select a Section:",
            ["Carbon Calculator",
             "Sustainability Dashboard",
             "Recommendations",
             "Reports",
             "Community Hub"]
        )

# Employee Features
def render_project_dashboard():
    st.title("AI Project Carbon Footprint Tracker")
    
    # Project selection
    project_data = generate_mock_project_data()
    project_name = st.selectbox("Select Project", project_data['Project'].tolist())
    
    # Metrics tabs
    tab1, tab2, tab3 = st.tabs(["Real-Time Metrics", "Stage Breakdown", "Historical Trends"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Carbon Emissions", "123 kg CO₂e", "-5%")
        with col2:
            st.metric("Energy Usage", "456 kWh", "+2%")
        with col3:
            st.metric("Water Usage", "789 L", "-3%")
    
    with tab2:
        stage_data = pd.DataFrame({
            'Stage': ['Training', 'Inference', 'Storage'],
            'Emissions': [45, 30, 25]
        })
        fig = px.bar(stage_data, x='Stage', y='Emissions',
                    title='Emissions by Stage (kg CO₂e)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        historical_data = generate_mock_emissions_data()
        fig = px.line(historical_data, x='Date', y='Emissions (kg CO₂e)',
                     title='Historical Emissions Trend')
        st.plotly_chart(fig, use_container_width=True)

def render_team_collaboration():
    st.title("Team Collaboration Dashboard")
    
    # Team overview
    team_data = generate_mock_team_data()
    
    # Team metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Team Size", str(len(team_data)), "+1 this month")
    with col2:
        st.metric("Active Projects", str(team_data['Projects'].sum()), "+2 this month")
    with col3:
        st.metric("Average Impact Score", f"{team_data['Impact Score'].mean():.1f}", "+5.2")
    
    # Team performance
    st.subheader("Team Performance")
    fig = px.bar(team_data, x='Member', y='Impact Score',
                 title='Team Impact Scores')
    st.plotly_chart(fig, use_container_width=True)
    
    # Project collaboration
    st.subheader("Project Collaboration")
    cols = st.columns(3)
    with cols[0]:
        st.number_input("Add team member to project", min_value=1, max_value=10, value=1)
    with cols[1]:
        st.selectbox("Select Project", ["Project A", "Project B", "Project C"])
    with cols[2]:
        st.button("Add to Project")
    
    # Knowledge sharing
    st.subheader("Knowledge Sharing")
    with st.expander("Share Best Practices"):
        st.text_area("Share your sustainable AI practices")
        st.button("Post")

def render_certifications():
    st.title("Certifications & Rewards")
    
    # Available certifications
    st.subheader("Available Certifications")
    cert_data = {
        "Green AI Pioneer": {"progress": 0.8, "requirements": "Reduce carbon footprint by 20%"},
        "Energy Optimizer": {"progress": 0.6, "requirements": "Implement 5 energy-saving optimizations"},
        "Sustainability Champion": {"progress": 0.4, "requirements": "Complete all green AI courses"}
    }
    
    for cert, data in cert_data.items():
        st.write(f"**{cert}**")
        st.write(f"Requirements: {data['requirements']}")
        st.progress(data['progress'])
    
    # Rewards and points
    st.subheader("Your Rewards")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Points", "1,234", "+56 this week")
    with col2:
        st.metric("Current Rank", "Gold", "↑ from Silver")
    
    # Redeem rewards
    st.subheader("Redeem Rewards")
    rewards = {
        "Cloud Credits": 500,
        "Training Course": 300,
        "Conference Ticket": 1000
    }
    
    selected_reward = st.selectbox("Select Reward", list(rewards.keys()))
    if st.button("Redeem"):
        st.success(f"Successfully redeemed {selected_reward}")

def render_impact_simulator():
    st.title("Impact Simulator")
    
    # Simulation parameters
    st.subheader("Scenario Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        model_size = st.slider("Model Size (GB)", 1, 100, 10)
        training_time = st.slider("Training Time (hours)", 1, 168, 24)
        batch_size = st.slider("Batch Size", 16, 512, 32)
    
    with col2:
        cloud_region = st.selectbox("Cloud Region", 
                                  ["US East", "US West", "EU Central", "Asia Pacific"])
        hardware_type = st.selectbox("Hardware Type",
                                   ["GPU V100", "GPU A100", "TPU v3", "TPU v4"])
        optimization = st.multiselect("Optimizations",
                                    ["Quantization", "Pruning", "Knowledge Distillation"])
    
    # Run simulation
    if st.button("Run Simulation"):
        # Mock simulation results
        base_emissions = model_size * training_time * 0.5
        optimized_emissions = base_emissions * (0.7 if optimization else 1.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline Emissions", f"{base_emissions:.1f} kg CO₂e")
        with col2:
            st.metric("Optimized Emissions", f"{optimized_emissions:.1f} kg CO₂e",
                     f"-{((base_emissions - optimized_emissions)/base_emissions*100):.1f}%")
        
        # Visualization
        timeline = pd.DataFrame({
            'Hour': range(training_time),
            'Baseline': [base_emissions/training_time] * training_time,
            'Optimized': [optimized_emissions/training_time] * training_time
        })
        
        fig = px.line(timeline, x='Hour', y=['Baseline', 'Optimized'],
                     title='Projected Emissions Over Time')
        st.plotly_chart(fig, use_container_width=True)

# Customer Features
def render_carbon_calculator():
    st.title("AI Carbon Footprint Calculator")
    
    with st.form("calculator_form"):
        st.subheader("Input Your AI Usage Details")
        
        # Model details
        model_size = st.slider("Model Size (GB)", 0, 100, 10)
        training_hours = st.number_input("Training Hours per Month", 0, 1000, 100)
        inference_calls = st.number_input("Monthly Inference Calls (thousands)", 0, 1000, 50)
        
        submitted = st.form_submit_button("Calculate Impact")
        
        if submitted:
            # Mock calculation
            total_emissions = (model_size * 0.1 + training_hours * 0.5 + inference_calls * 0.01)
            st.success(f"Estimated Monthly Carbon Footprint: {total_emissions:.2f} kg CO₂e")
            
            # Scenario analysis
            st.subheader("Impact Reduction Scenarios")
            scenarios = pd.DataFrame({
                'Scenario': ['Current', 'Optimized Model', 'Green Infrastructure'],
                'Emissions': [total_emissions, total_emissions * 0.7, total_emissions * 0.4]
            })
            fig = px.bar(scenarios, x='Scenario', y='Emissions',
                        title='Potential Emissions by Scenario')
            st.plotly_chart(fig, use_container_width=True)



def render_training_hub():
    st.title("Training Hub")
    
    st.subheader("Sustainable AI Practices")
    st.write("Learn about best practices for reducing the environmental impact of AI.")
    
    # List of resources
    resources = {
        "AI for Earth": "https://www.microsoft.com/en-us/sustainability/emissions-impact-dashboard",
        "Green AI Course": "https://www.coursera.org/learn/green-ai",
        "Sustainable Machine Learning": "https://www.udemy.com/course/sustainable-machine-learning/"
    }
    
    for title, link in resources.items():
        st.write(f"- [{title}]({link})")
    
    st.subheader("Track Your Learning Progress")
    
    # Learning progress tracking
    courses = st.multiselect("Select Courses Completed", list(resources.keys()))
    
    if st.button("Submit Progress"):
        st.success("You have completed: " + ", ".join(courses))
        # Here you could implement logic to save this progress to a database or session state

    st.subheader("Upcoming Training Sessions")
    
    # Mock upcoming sessions
    sessions = pd.DataFrame({
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "Session": ["Sustainable AI Best Practices", "Energy-Efficient Model Training", "Carbon Footprint Reduction in AI"]
    })
    
    st.table(sessions)



def render_development_tools():
    st.title("Development Tools")
    
    st.subheader("Model Optimization Tools")
    
    # Model Optimization Options
    optimization_options = st.multiselect(
        "Select Optimization Techniques",
        ["Quantization", "Pruning", "Knowledge Distillation", "Mixed Precision Training"]
    )
    
    if st.button("Run Optimization"):
        st.success("Running optimizations: " + ", ".join(optimization_options))
        # Here you would implement the logic to run these optimizations
        st.write("Optimizations applied successfully!")

    st.subheader("Code Snippets for Efficient Energy Usage")
    
    code_snippet = """
    # Example of using mixed precision training in PyTorch
    import torch
    from torch.cuda.amp import autocast, GradScaler

    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()

    for data, target in dataloader:
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """
    
    st.code(code_snippet, language='python')
    
    st.subheader("Useful Libraries and Resources")
    st.write("1. [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)")
    st.write("2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)")
    st.write("3. [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)")
# Customer Features
def render_sustainability_dashboard():
    st.title("Sustainability Dashboard")
    
    st.subheader("Overview of Sustainability Metrics")
    metrics_data = {
        "Total Emissions (kg CO₂e)": np.random.randint(5000, 20000),
        "Energy Consumption (kWh)": np.random.randint(30000, 100000),
        "Water Usage (L)": np.random.randint(100000, 500000),
        "Sustainable Projects": np.random.randint(5, 20)
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emissions", f"{metrics_data['Total Emissions (kg CO₂e)']}")
    with col2:
        st.metric("Energy Consumption", f"{metrics_data['Energy Consumption (kWh']}")
    with col3:
        st.metric("Water Usage", f"{metrics_data['Water Usage (L)']}")
    with col4:
        st.metric("Sustainable Projects", f"{metrics_data['Sustainable Projects']}")
    
    st.subheader("Emissions Over Time")
    historical_data = generate_mock_emissions_data()
    fig = px.line(historical_data, x='Date', y='Emissions (kg CO₂e)',
                  title='Historical Emissions Trend')
    st.plotly_chart(fig, use_container_width=True)

def render_recommendations():
    st.title("Recommendations")
    
    st.subheader("Personalized Recommendations")
    recommendations = [
        "Implement energy-efficient algorithms.",
        "Use cloud services with renewable energy sources.",
        "Optimize data processing to reduce resource usage.",
        "Consider model distillation for smaller model sizes."
    ]
    
    for rec in recommendations:
        st.write(f"- {rec}")

def render_reports():
    st.title("Reports")
    
    st.subheader("Generate Reports")
    st.write("Select the parameters for your report:")
    
    report_type = st.selectbox("Select Report Type", ["Monthly Emissions", "Project Impact", "Resource Usage"])
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.today())
    
    if st.button("Generate Report"):
        st.success(f"Report '{report_type}' generated from {start_date} to {end_date}.")
        # Here you would implement logic to generate and display the report

def render_community_hub():
    st.title("Community Hub")
    
    st.subheader("Connect with Other Users")
    st.write("Join discussions and share best practices.")
    
    with st.expander("Discussion Forum"):
        st.text_area("Share your thoughts or ask questions:")
        if st.button("Post"):
            st.success("Your message has been posted!")
    
    st.subheader("Upcoming Community Events")
    events = pd.DataFrame({
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "Event": ["Sustainable AI Workshop", "Green Technology Summit", "Carbon Footprint Awareness Day"]
    })
    
    st.table(events)


def show_energy_monitoring():
    """Display energy monitoring charts"""
    energy_data = pd.DataFrame({
        'hour': range(24),
        'consumption': np.random.normal(50, 10, 24)
    })
    fig = px.line(energy_data, x='hour', y='consumption',
                  title='24-Hour Energy Consumption')
    st.plotly_chart(fig)

def show_building_controls():
    """Display building control interface"""
    st.slider("Temperature Setting", min_value=18, max_value=28, value=22)
    st.slider("Lighting Level", min_value=0, max_value=100, value=70)
    st.checkbox("Enable Smart Mode", value=True)

def show_smart_building_dashboard():
    """Display the smart building dashboard."""
    st.title("Smart Building Controls")
    show_energy_monitoring()
    show_building_controls()


def render_carbon_calculator():
    st.title("AI Carbon Footprint Calculator")
    
    tab1, tab2 = st.tabs(["Basic Calculator", "Advanced Analysis"])
    
    with tab1:
        with st.form("calculator_form"):
            st.subheader("Input Your AI Usage Details")
            
            col1, col2 = st.columns(2)
            with col1:
                model_size = st.slider("Model Size (GB)", 0, 100, 10)
                training_hours = st.number_input("Training Hours per Month", 0, 1000, 100)
            with col2:
                inference_calls = st.number_input("Monthly Inference Calls (thousands)", 0, 1000, 50)
                hardware_type = st.selectbox("Hardware Type", ["CPU", "GPU", "TPU"])
            
            submitted = st.form_submit_button("Calculate Impact")
            
            if submitted:
                # Enhanced calculation with hardware factors
                hardware_factors = {"CPU": 1.0, "GPU": 1.5, "TPU": 0.8}
                total_emissions = (model_size * 0.1 + training_hours * 0.5 + inference_calls * 0.01) * hardware_factors[hardware_type]
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monthly Carbon Footprint", f"{total_emissions:.2f} kg CO₂e")
                with col2:
                    st.metric("Annual Projection", f"{(total_emissions * 12):.2f} kg CO₂e")
                with col3:
                    trees_needed = total_emissions * 12 / 21  # Average tree absorbs 21kg CO2 annually
                    st.metric("Trees Needed to Offset", f"{trees_needed:.1f} trees")

    with tab2:
        st.subheader("Compare with Industry Standards")
        industry_data = pd.DataFrame({
            'Industry': ['Healthcare', 'Finance', 'E-commerce', 'Your Usage'],
            'Emissions': [150, 200, 180, total_emissions if 'total_emissions' in locals() else 0]
        })
        fig = px.bar(industry_data, x='Industry', y='Emissions',
                    title='Monthly Emissions Comparison (kg CO₂e)',
                    color='Industry',
                    color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
        st.plotly_chart(fig, use_container_width=True)

def render_sustainability_dashboard():
    st.title("Sustainability Dashboard")
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        period = st.select_slider("Select Time Period", 
                                options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                                value="Monthly")
    with col2:
        st.metric("Sustainability Score", "87/100", "+5 from last period")
    
    # Interactive metrics with drilldown capability
    metrics_data = {
        "Total Emissions": {"value": np.random.randint(5000, 20000), "unit": "kg CO₂e", "trend": -5},
        "Energy Consumption": {"value": np.random.randint(30000, 100000), "unit": "kWh", "trend": -3},
        "Water Usage": {"value": np.random.randint(100000, 500000), "unit": "L", "trend": -8},
        "Resource Efficiency": {"value": np.random.randint(75, 95), "unit": "%", "trend": +2}
    }
    
    cols = st.columns(len(metrics_data))
    for i, (metric, data) in enumerate(metrics_data.items()):
        with cols[i]:
            st.metric(
                metric,
                f"{data['value']} {data['unit']}",
                f"{data['trend']:+d}%"
            )
    
    # Interactive trends visualization
    tab1, tab2 = st.tabs(["Trends Analysis", "Resource Breakdown"])
    
    with tab1:
        # Generate mock historical data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        historical_data = pd.DataFrame({
            'Date': dates,
            'Emissions': np.random.normal(100, 10, 30),
            'Energy': np.random.normal(500, 50, 30),
            'Water': np.random.normal(200, 20, 30)
        })
        
        metric_to_plot = st.selectbox("Select Metric to Visualize", 
                                    ['Emissions', 'Energy', 'Water'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data[metric_to_plot],
            mode='lines+markers',
            name=metric_to_plot,
            line=dict(color='#2ecc71')
        ))
        fig.update_layout(title=f'{metric_to_plot} Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Resource usage breakdown
        resources_breakdown = pd.DataFrame({
            'Resource': ['Computing', 'Storage', 'Networking', 'Cooling'],
            'Usage': np.random.uniform(10, 30, 4)
        })
        fig = px.pie(resources_breakdown, values='Usage', names='Resource',
                    title='Resource Usage Breakdown',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

def render_recommendations():
    st.title("Smart Recommendations")
    
    # Calculate mock sustainability score
    sustainability_score = np.random.randint(60, 95)
    
    # Progress indicator
    st.progress(sustainability_score/100)
    st.write(f"Your Sustainability Score: {sustainability_score}/100")
    
    # Priority recommendations
    st.subheader("Priority Actions")
    priorities = {
        "High": ["Switch to renewable energy providers", "Implement model compression"],
        "Medium": ["Optimize batch processing", "Update hardware efficiency"],
        "Low": ["Review data storage practices", "Consider edge computing"]
    }
    
    for priority, actions in priorities.items():
        with st.expander(f"{priority} Priority"):
            for action in actions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {action}")
                with col2:
                    st.button("Mark Complete", key=f"btn_{action}")
    
    # Impact calculator
    st.subheader("Potential Impact Calculator")
    selected_action = st.selectbox(
        "Select an action to calculate potential impact",
        ["Model compression", "Renewable energy", "Edge computing", "Efficient hardware"]
    )
    
    if st.button("Calculate Potential Impact"):
        impact = {
            "emissions_reduction": np.random.randint(10, 30),
            "cost_savings": np.random.randint(1000, 5000),
            "efficiency_gain": np.random.randint(15, 40)
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emissions Reduction", f"{impact['emissions_reduction']}%")
        with col2:
            st.metric("Cost Savings", f"${impact['cost_savings']}")
        with col3:
            st.metric("Efficiency Gain", f"{impact['efficiency_gain']}%")

def render_reports():
    st.title("Advanced Analytics & Reports")
    
    # Report configuration
    col1, col2 = st.columns(2)
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            ["Environmental Impact", "Resource Optimization", "Cost Analysis", "Custom Report"]
        )
    with col2:
        report_format = st.selectbox("Export Format", ["PDF", "Excel", "Interactive Dashboard"])
    
    # Date range with presets
    date_preset = st.radio(
        "Select Time Range",
        ["Last 30 Days", "Last Quarter", "Year to Date", "Custom Range"],
        horizontal=True
    )
    
    if date_preset == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.today() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.today())
    
    # Report components selection
    st.subheader("Report Components")
    components = st.multiselect(
        "Select Components to Include",
        ["Emissions Analysis", "Cost Breakdown", "Resource Usage", "Optimization Opportunities",
         "Comparative Analysis", "Forecasting", "Recommendations"],
        default=["Emissions Analysis", "Resource Usage"]
    )
    
    if st.button("Generate Report"):
        # Show success message directly without progress animation
        st.success("Report generated successfully!")
        
        # Preview section
        st.subheader("Report Preview")
        tab1, tab2 = st.tabs(["Summary", "Detailed Analysis"])
        
        with tab1:
            st.write("### Key Findings")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Impact Reduction", "23%", "+5%")
                st.metric("Cost Savings", "$12,500", "+$2,300")
            with col2:
                st.metric("Resource Efficiency", "87%", "+12%")
                st.metric("Carbon Credits Earned", "45", "+15")

def render_community_hub():
    st.title("Sustainability Community Hub")
    
    # User profile and achievements
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        # Instead of using image, we'll use an emoji and text
        st.markdown("👤")
        st.write("User Profile")
    with col2:
        st.subheader("Your Sustainability Profile")
        st.write("Level: Sustainability Champion")
        st.write("Points: 1,250")
        st.progress(0.75)
    with col3:
        st.metric("Community Rank", "#42", "↑ 5")
    
    # Community challenges
    st.subheader("Active Community Challenges")
    challenges = [
        {"name": "30-Day Green AI", "participants": 156, "days_left": 12},
        {"name": "Zero-Carbon Week", "participants": 89, "days_left": 5},
        {"name": "Resource Optimization", "participants": 234, "days_left": 18}
    ]
    
    for challenge in challenges:
        with st.expander(f"{challenge['name']} Challenge"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"🏃 {challenge['participants']} Participants")
            with col2:
                st.write(f"⏳ {challenge['days_left']} Days Left")
            with col3:
                st.button("Join Challenge", key=f"join_{challenge['name']}")
    
    # Knowledge sharing
    st.subheader("Knowledge Exchange")
    tab1, tab2 = st.tabs(["Share Insights", "Browse Topics"])
    
    with tab1:
        st.text_area("Share your sustainability practices or ask questions",
                    height=100)
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Category", ["Best Practices", "Questions", "Success Stories"])
        with col2:
            st.multiselect("Tags", ["AI", "Energy", "Optimization", "Resources"])
        st.button("Post")
    
    with tab2:
        topics = [
            {"title": "Reduced training time by 40%", "votes": 45, "comments": 12},
            {"title": "Zero-carbon deployment strategy", "votes": 38, "comments": 8},
            {"title": "Resource optimization tips", "votes": 29, "comments": 15}
        ]
        
        for topic in topics:
            st.write(f"### {topic['title']}")
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"↑ {topic['votes']}")
            with col2:
                st.write(f"💬 {topic['comments']} comments")
            st.divider()

# Main app
def main():
    if not st.session_state.logged_in:
        login_page()
        return
    
    section = sidebar_nav()
    
    # Employee sections
    if st.session_state.user_type == "employee":
        if section == "Project Dashboard":
            render_project_dashboard()
        elif section == "Development Tools":
            render_development_tools()
        elif section == "Team Collaboration":
            render_team_collaboration()
        elif section == "Certifications & Rewards":
            render_certifications()
        elif section == "Smart Building":
            show_smart_building_dashboard()
        elif section == "Impact Simulator":
            render_impact_simulator()
        elif section == "Training Hub":
            render_training_hub()
    
    # Customer sections
    else:
        if section == "Carbon Calculator":
            render_carbon_calculator()
        elif section == "Sustainability Dashboard":
            render_sustainability_dashboard()
        elif section == "Recommendations":
            render_recommendations()
        elif section == "Reports":
            render_reports()
        elif section == "Community Hub":
            render_community_hub()

    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.experimental_user()

if __name__ == "__main__":
    main()