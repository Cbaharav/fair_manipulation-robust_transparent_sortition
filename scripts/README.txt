SOLVING RAW PROBABILITIES         (These can run on your machine, not too time-intensive)


paper_data_analysis.py can run quota-based leximin and nash on any instance to get raw selection probabilities

    **Solver-wise, we should emulate approach to solving nash for FF**
    **Solver-wise, we should emulate approach to solving leximin for infinity norm**

analysis.py analyzes LEGACY, the previous SOTA algorithm 

Both of these scripts come from the paper Fair Algorithms for Selecting Citizens' Assemblies. That code base can be found here: 
https://github.com/pgoelz/citizensassemblies-replication




TRANSPARENCY ANALYSIS           (This analysis is really fast, can run easily on your machine)

paper_data_analysis.py can run rounding for transparency analysis.

paper_data_visualization.py can create plots replicating those in the transparency paper. 


Both of these scripts come from the paper Fair Sortition Made Transparent. Here is the code from that paper: https://github.com/baileyflanigan/fair_sortition_made_transparent





MANIPULATION ANALYSIS.          (I would recommend running this analysis on a cluster, if we do decide to do it)

vary_pool.py (which calls strategies.py) contains the framework for testing different kinds of manipulation strategies, I think including parallelization for speedups. 
The only difference is that this code tests the continuous (and not quota) version of the selection algorithm. Thus, the quota-based algorithms above must be plugged into the code.



Both of these scripts come from the paper Manipulation-Robust Citizens' Assembly Selection. Here is the code from that paper: https://drive.google.com/file/d/1F8NmYexjmJit7mAWVbIsO2o1Svoxa2OR/view

