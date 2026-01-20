import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./results_cmnd_ip.csv", index_col=False)
# df = pd.read_csv('./results_cmnd_itp.csv', index_col=False)
# print(df)
# df = pd.read_csv('./cmnd-preliminary.csv')
df.columns = [c.replace(" ", "_") for c in df.columns]
print(df.Method)
df["Method"] = df["Method"].replace(["AdaptiveInit"], "Adaptive init")
df["Method"] = df["Method"].replace(["DSP"], "DSP")
df["Method"] = df["Method"].replace(["CuratedDSP"], "Curated DSP")
df["Method"] = df["Method"].replace(["StaticInit"], "Static init")
# df['Method'] = df['Method'].replace(['1_1_0_0_0_0'],'No reuse')
df_new = df[
    df.Method.isin(("No reuse", "Adaptive init", "Static init", "DSP", "Curated DSP"))
]  # ["no init", "active LP + tech 2", "no LP + tech 2"]]
print(df.columns)

legend_order = ["DSP", "Curated DSP", "Static init", "Adaptive init"]
# legend_order = ['No reuse', 'DSP', 'Curated DSP', 'Static init', 'Adaptive init']

sns.ecdfplot(
    data=df_new, x="Total_times_average", hue="Method", hue_order=legend_order
).set(
    xlabel="Total time average",
    ylabel="Proportion of instances",
    # title="IP time distribution",
)
# sns.ecdfplot(data=df_new, x='total_time_average', hue='Method').set(xlabel="Total time average", ylabel="Proportion of instances", title='IP time distribution')
plt.savefig("cmnd_cdf_ip.png")
