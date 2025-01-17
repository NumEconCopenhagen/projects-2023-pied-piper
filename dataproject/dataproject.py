import matplotlib.pyplot as plt

# 1. Defining our Lorenz diagram in a function so that we call in the Jupyter file and add a slider widget
def L_Diagrams(Empty_Lists, Year, Cum_Num):
    """
    This function creates a Lorenz Diagram for a given year, and plots it against the 45 degree line.

    The function takes three arguments:
    Empty_Lists: A list of lists, where each list is a list of the cumulative share of income for each decile for a given year.
    Year: An integer between 2010 and 2021, which is the year we want to plot the Lorenz Diagram for.
    Cum_Num: A list of numbers from 0 to 100, which is the cumulative share of population.
    
    The function returns a Lorenz Diagram for the given year.
    """

    # a. Creating the figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # b. Plotting the Lorenz Curve and the 45 degree line
    List_Choosing = Empty_Lists[Year - 2010]
    ax.plot(Cum_Num,List_Choosing, 'o-', label="Lorenz Curve")
    ax.plot(Cum_Num,Cum_Num, label="45 degree line")

    # c. Setting title and legend
    ax.set_title('Plot 1: Lorenz Diagram')
    ax.set_xlabel('Share of population')
    ax.set_ylabel('Accumulative disposable income');
    ax.legend(['Lorenz Curve', '45 degree line'])

    # d. Adding a grid
    ax.grid(True)

    # e. Setting limits on each axis
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # f. Creating loop to show the values on our Lorenz Curve
    for x, y in zip(Cum_Num, List_Choosing):
        # i. Setting the label and the decimals
        Values = f"({x:.2f}, {y:.2f})"
        ax.annotate(Values, # Choosing the values to display
                    xy=(x, y), 
                    textcoords="offset points", # Choosing where to show the pount
                    xytext=(5,5), # Choosing the distance from text to points
                    ha='left') # Setting alignment


# 2. Defining our average income diagram
def Decile_Comp(decils, decil_list, income_decil, PctG_GDP, plot_GDP_growth=True, plot_2pct_line=True):
    """
    This function creates a plot of the average disposable income for each decile for a given year.
    
    The function the following arguments:
    decils: A list of integers from 1 to 10, which is the deciles we want to plot.
    decil_list: A list of strings, which is the names of the deciles we want to plot.
    income_decil: A dataframe, which is the dataframe we want to use.
    PctG_GDP: A dataframe, which is the dataframe we want to use.
    plot_GDP_growth: A boolean, which is True if we want to plot the GDP growth and False if we don't.
    plot_2pct_line: A boolean, which is True if we want to plot the 2% line and False if we don't.
    
    The function returns a plot of the average disposable income for each decile for a given year.
    """

    # a. Making the plot  
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # b. Using loop to plot the average disposable income for each decile
    for d in decils:
        # i. Finding the index of the decile
        i = decil_list.index(d)

        # ii. Choosing what income to plot
        income = income_decil.iloc[i]

        # iii. Plotting the income
        ax.plot(income, label=decil_list[i])

        # iv. Setting title and labels
        ax.set_title('Plot 2: Decile income comparison')
        ax.set_ylabel('Average disposable income (DKK)')
        ax.set_xlabel('Year')

        # v. Setting limits on each axis
        ax.set_xlim(0, 11)

    # c. Creating a second y-axis for the GDP growth and 2% line    
    ax2 = ax.twinx()

    # d. Plotting the GDP growth
    if plot_GDP_growth:
        ax2.plot(PctG_GDP['Percentage Growth'], color='green', label='GDP growth, %', linestyle='-.')
    
    # e. Plotting the 2% line
    if plot_2pct_line:
        ax2.plot(PctG_GDP['2pct line'], color='red', label='2% line', linestyle='--')
    
    # f. Setting title, labels and grid
    ax.legend()
    ax2.set_ylabel('Percentage Growth') 
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.00))
    ax.grid(True)
    
    plt.show()
