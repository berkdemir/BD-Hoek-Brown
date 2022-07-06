"""
Hoek-Brown Calculation by Berk Demir
"""

from math import exp, asin, degrees, sin, cos, radians, log10
import numpy as np
from numpy import power as pow
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

favicon = Image.open("images/favicon.ico")

st.set_page_config(
    page_title="BD Hoek Brown",
    page_icon=":pick:",
    layout="wide",
    initial_sidebar_state="expanded",
)

a = st.sidebar
BD_im = Image.open("images/BDTunnelTools.png")
a.image(BD_im)


class HoekBrown:
    def __init__(
        self,
        UCS,
        GSI,
        mi,
        D,
        E_Method,
        sigma3_method,
        UW,
        H,
        MR=None,
        Name="Rock",
        Ei=None,
    ):
        self.UCS = UCS
        self.GSI = GSI
        self.mi = mi
        self.D = D
        self.E_Method = E_Method
        self.sigma3_method = sigma3_method
        self.UW = UW
        self.H = H
        self.MR = MR
        self.Name = Name
        self.Ei = Ei
        self.HBParameters()
        self.results = {
            r"$m_b$": round(self.mb, 3),
            r"$s$": round(self.s, 5),
            r"$\alpha$": round(self.alpha, 3),
            r"$\sigma_{cm}\/(MPa)$": round(self.sigmacm, 2),
            r"$\sigma_{3}\/(MPa)$": round(self.sigma3, 2),
            r"$\sigma_{t}\/(MPa)$": round(self.sigmat, 3),
            r"$E_{rm}\/(MPa)$": round(self.Erm),
            r"$c\/(kPa)$": round(self.c * 1000),
            r"$\varphi\/(°)$": round(self.fi, 1),
        }

        self.inputs = {
            r"  $UCS$  ": round(self.UCS, 1),
            r"  $GSI$  ": round(self.GSI, 0),
            r"   $m_i$   ": round(self.mi, 0),
            r"   $D$   ": round(self.D, 2),
            r"  $Modulus\/\/Calculation$  ": self.E_Method,
            r" $\sigma_{3}\/\/Method$  ": self.sigma3_method,
        }

        if self.UW != None:
            self.inputs[r"  $\gamma\/(kN/m^3)$  "] = self.UW
        if self.H != None:
            self.inputs[r"  $Depth$  "] = self.H
        if self.MR != None:
            self.inputs[r"  $MR$  "] = self.MR
        self.HB_Figure()

    def HBParameters(self):
        self.mb = self.mi * exp((self.GSI - 100) / (28 - 14 * self.D))
        self.s = exp((self.GSI - 100) / (9 - 3 * self.D))
        self.alpha = 0.5 + (exp(-self.GSI / 15) - exp(-20 / 3)) / 6
        self.sigmacm_calc()
        self.sigmat_calc()
        self.MC()
        self.E_calc()
        self.Additional_Parameters()

    def sigmacm_calc(self):
        """
        Calculates rock mass strength (sigmacm) in MPa
        """
        self.sigmacm = (
            self.UCS
            * (self.mb + 4 * self.s - self.alpha * (self.mb - 8 * self.s))
            * pow(self.mb / 4 + self.s, self.alpha - 1)
            / (2 * (1 + self.alpha) * (2 + self.alpha))
        )
        self.sigma3_calc()

    def sigma3_calc(self):
        """
        Calculates lateral pressure (sigma3) in MPa.
        """
        if self.sigma3_method == "Tunnel":
            self.sigma3 = (
                0.47
                * pow(self.sigmacm / (self.UW * self.H * 0.001), -0.94)
                * self.sigmacm
            )
        elif self.sigma3_method == "Slope":
            self.sigma3 = (
                0.72
                * pow(self.sigmacm / (self.UW * self.H * 0.001), -0.91)
                * self.sigmacm
            )
        elif self.sigma3_method == "General":
            self.sigma3 = self.UCS / 4

    def sigmat_calc(self):
        """
        Calculates tensile strength (sigmat) in MPa.
        """
        self.sigmat = -self.s * self.UCS / self.mb

    def MC(self):
        """
        Calculates Mohr-Coulomb parameters c (in MPa) and fi (in deg).
        """
        sigma3n = self.sigma3 / self.UCS
        self.fi = (
            6
            * self.alpha
            * self.mb
            * pow(self.s + self.mb * sigma3n, self.alpha - 1)
            / (
                2 * (1 + self.alpha) * (2 + self.alpha)
                + 6
                * self.alpha
                * self.mb
                * pow(self.s + self.mb * sigma3n, self.alpha - 1)
            )
        )
        self.fi = degrees(asin(self.fi))

        self.c = (
            self.UCS
            * ((1 + self.alpha * 2) * self.s + (1 - self.alpha) * self.mb * sigma3n)
            * pow(self.s + self.mb * sigma3n, self.alpha - 1)
            / (
                (1 + self.alpha)
                * (2 + self.alpha)
                * pow(
                    1
                    + (
                        6
                        * self.alpha
                        * self.mb
                        * pow(self.s + self.mb * sigma3n, self.alpha - 1)
                    )
                    / ((1 + self.alpha) * (2 + self.alpha)),
                    0.5,
                )
            )
        )

    def E_calc(self):
        """
        Calculates rock mass modulus in MPa.
        """
        if self.E_Method == "Generalized Hoek & Diederichs (2006)":
            if not self.Ei:
                self.Ei = self.MR * self.UCS
            self.Erm = self.Ei * (
                0.02 + (1 - self.D / 2) / (1 + exp((60 + 15 * self.D - self.GSI) / 11))
            )
        elif self.E_Method == "Simplified Hoek & Diederichs (2006)":
            self.Erm = 100000 * (
                (1 - self.D / 2) / (1 + exp((75 + 25 * self.D - self.GSI) / 11))
            )
        elif self.E_Method == "Hoek, Carranza-Torres, Corkum (2002)":
            if self.UCS <= 100:
                self.Erm = (
                    1000
                    * (1 - self.D / 2)
                    * pow(self.UCS / 100, 0.5)
                    * pow(10, (self.GSI - 10) / 40)
                )
            else:
                self.Erm = 1000 * (1 - self.D / 2) * pow(10, (self.GSI - 10) / 40)
        elif self.E_Method == "Yang (2006) / AASHTO":
            if not self.Ei:
                self.Ei = self.MR * self.UCS
            self.Erm = self.Ei * 0.01 * exp(self.GSI / 21.7)
        else:
            self.Erm = None  # User will change it manually by accessing self.Erm value.

    def Additional_Parameters(self):

        self.RMR = self.GSI + 5
        self.Q = pow(10, (self.RMR - 50) / 15)
        self.VP = 3.5 + log10(self.Q * self.UCS / 100)
        self.VS = (
            0.7858
            - 1.2344 * self.VP
            + 0.7949 * self.VP ** 2
            - 0.1238 * self.VP ** 3
            + 0.0064 * self.VP ** 4
        )
        self.add_parameters = {
            "Shear Wave Velocity by Brocher 2005 (m/s)": round(self.VS * 1000),
            "Poisson's Ratio by Brocher 2005": round(
                0.8835
                - 0.315 * self.VP
                + 0.0491 * self.VP ** 2
                - 0.0024 * self.VP ** 3,
                2,
            ),
            "Shear Wave Velocity by Cha 2006 (m/s)": round(
                (self.RMR + 10.021) / 0.0362
            ),
            "Residual GSI by Alejano 2012": round(17.34 * exp(0.0107 * self.GSI), 1),
            "Residual GSI by Cai 2007": round(self.GSI * exp(-0.0134 * self.GSI), 1),
            "Dilation Angle by Alejano 2010 (°)": round(
                self.fi * (5 * self.GSI - 125) / 1000, 1
            ),
            "Poisson's Ratio by Aydan 1993": round(
                0.25 * (1 + exp(-0.2 * self.UCS)), 2
            ),
            "Poisson's Ratio by Vasarhelyi 2009": round(
                -0.002 * self.GSI - 0.003 * self.mi + 0.457, 2
            ),
            "Unit Weight by Aydan 1993 (kN/m³)": round(
                10 * (1 + 0.8 * pow(self.UCS, 0.14)), 1
            ),
        }

    def HB_Figure(self):
        sigma3_array = np.linspace(self.sigmat, self.sigma3, 1000)
        sigma1_array = sigma3_array + self.UCS * pow(
            self.mb * sigma3_array / self.UCS + self.s, self.alpha
        )
        sigma1_MC_array = 2 * self.c * cos(radians(self.fi)) / (
            1 - sin(radians(self.fi))
        ) + (1 + sin(radians(self.fi))) * sigma3_array / (1 - sin(radians(self.fi)))
        sigma1_MC_array[0] = 0
        sigma1_array[sigma1_array < 0] = 0

        self.fig, ax = plt.subplots(dpi=300, figsize=(10, 5))

        ax.plot(sigma3_array, sigma1_array, label="Hoek-Brown", color="black")
        ax.plot(sigma3_array, sigma1_MC_array, label="Mohr-Coulomb Fit", color="red")
        # ax.spines['left'].set_position('center')
        ax.margins(0)
        ax.set_title("Hoek Brown Analysis for " + self.Name, fontweight="bold")
        ax.set_xlabel(r"$\sigma_3$ - Minor Stress (MPa)")
        ax.set_ylabel(r"$\sigma_1$ - Major Stress (MPa)")
        ax.plot([], [], " ", label="github.com/berkdemir")
        ax.legend(loc=4, fontsize="small")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.fill_between(
            sigma3_array, sigma1_array, sigma1_MC_array, alpha=0.1, color="black"
        )
        val1 = [i for i in self.results.keys()]
        val2 = [r"$\mathbf{Results}$"]
        val3 = [[i for i in self.results.values()] for r in range(1)]
        table1 = ax.table(
            cellText=val3,
            rowLabels=val2,
            colLabels=val1,
            colLoc="center",
            cellLoc="center",
            rowLoc="center",
            loc="bottom",
            bbox=[0.1, -0.55, 0.85, 0.15],
        )
        val4 = [i for i in self.inputs.keys()]
        val5 = [r"  $\mathbf{Inputs}$  "]
        val6 = [[i for i in self.inputs.values()] for r in range(1)]
        table2 = ax.table(
            cellText=val6,
            rowLabels=val5,
            colLabels=val4,
            colLoc="center",
            cellLoc="center",
            rowLoc="center",
            loc="bottom",
            bbox=[0.1, -0.35, 0.9, 0.15],
        )

        [t.auto_set_font_size(False) for t in [table1, table2]]
        [t.set_fontsize(9) for t in [table1, table2]]

        table2.auto_set_column_width(col=list(range(len(self.inputs.keys()))))
        table1.auto_set_column_width(col=list(range(len(self.results.keys()))))


def streamlitHoekBrown():
    st.subheader("Hoek Brown Analysis")
    a = st.sidebar

    a.title("Information and Settings")
    a.subheader("Settings:")
    Name = a.text_input("Rock layer's name", value="Rock")

    a.subheader("Theory:")
    a.markdown(
        "[Link to Theory](https://berkdemir.github.io/2021/09/18/Hoek-Brown-Model/)"
    )

    a.subheader("Interview with Evert Hoek:")
    a.markdown(
        "[Link for Interview](https://github.com/berkdemir/berkdemir.github.io/blob/main/images/Hoek%20-%20Interview.pdf)"
    )

    a.subheader("Reference:")
    a.write(
        "Following reference should be used for commercial or academical use: Berk Demir (2021) Hoek Brown Tool at github.com/berkdemir"
    )

    a.markdown(
        '<font size="1.5">Disclaimer: This is a personal project and not endorsed by any other parties. All responsibility with the use of this tool lies with the user. Proper QA should be performed to use in any project.</font>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 2])

    with cols[0]:
        st.markdown("**General Parameters**")
        UCS = st.number_input(
            "Uniaxial Compressive Strength (MPa)", value=30.0, min_value=0.01
        )
        GSI = st.number_input(
            "Geological Strength Index (GSI)",
            value=60,
            min_value=1,
            max_value=100,
        )
        mi = st.number_input("Material constant for intact rock", value=10, min_value=1)
        D = st.number_input(
            "Disturbance Factor", value=0.0, min_value=0.0, max_value=1.0
        )
        st.markdown("**Rock Mass Modulus Calculations**")
        E_Method = st.selectbox(
            "Calculation method",
            (
                "Generalized Hoek & Diederichs (2006)",
                "Simplified Hoek & Diederichs (2006)",
                "Hoek, Carranza-Torres, Corkum (2002)",
                "Yang (2006) / AASHTO",
            ),
        )
        if (
            E_Method == "Generalized Hoek & Diederichs (2006)"
            or E_Method == "Yang (2006) / AASHTO"
        ):
            Ei_method = st.selectbox(
                "Ei calculation method", ("Using MR (Modulus Ratio)", "Manual Input")
            )
            if Ei_method == "Using MR (Modulus Ratio)":
                MR = st.number_input("Modulus Ratio (Ei/UCS)", value=350, min_value=1)
                Ei = None
            else:
                Ei = st.number_input(
                    "Intact modulus elasticity of rock", value=1000, min_value=1
                )
                MR = None
        else:
            MR = None
            Ei = None

    with cols[1]:
        st.markdown("**Equivalent Mohr-Coulomb Calculations**")
        sigma3_method = st.selectbox(
            "Lateral pressure calculation method", ("Tunnel", "Slope", "General")
        )
        if sigma3_method != "General":
            UW = st.number_input(
                "Unit weight of rock (kN/m3)", value=23.5, min_value=1.0
            )
            H = st.number_input("Tunnel or slope depth (m)", value=30, min_value=1)
        else:
            UW = None
            H = None

        st.markdown("**Additional Parameters**")
        RockDict = {
            "Agglomerate": {
                "MR": 500,
                "MRSTD": 100,
                "Type": "Igneous",
                "mi": 19,
                "miSTD": 3,
            },
            "Amphibolites": {
                "MR": 450,
                "MRSTD": 50,
                "Type": "Metamorphic",
                "mi": 26,
                "miSTD": 6,
            },
            "Andesite": {
                "MR": 400,
                "MRSTD": 100,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 5,
            },
            "Anhydrite": {
                "MR": 350,
                "MRSTD": 0,
                "Type": "Sedimentary",
                "mi": 12,
                "miSTD": 2,
            },
            "Basalt": {
                "MR": 350,
                "MRSTD": 100,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 5,
            },
            "Breccia-I": {
                "MR": 500,
                "MRSTD": 0,
                "Type": "Igneous",
                "mi": 19,
                "miSTD": 5,
            },
            "Breccia-S": {
                "MR": 290,
                "MRSTD": 60,
                "Type": "Sedimentary",
                "mi": 19,
                "miSTD": 5,
            },
            "Chalk": {
                "MR": 1000,
                "MRSTD": 0,
                "Type": "Sedimentary",
                "mi": 7,
                "miSTD": 2,
            },
            "Claystones": {
                "MR": 250,
                "MRSTD": 50,
                "Type": "Sedimentary",
                "mi": 4,
                "miSTD": 2,
            },
            "Conglomerates": {
                "MR": 350,
                "MRSTD": 50,
                "Type": "Sedimentary",
                "mi": 21,
                "miSTD": 3,
            },
            "Dacite": {
                "MR": 400,
                "MRSTD": 50,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 3,
            },
            "Diabase": {
                "MR": 325,
                "MRSTD": 25,
                "Type": "Igneous",
                "mi": 15,
                "miSTD": 5,
            },
            "Diorite": {
                "MR": 325,
                "MRSTD": 25,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 5,
            },
            "Dolerite": {
                "MR": 350,
                "MRSTD": 50,
                "Type": "Igneous",
                "mi": 12,
                "miSTD": 3,
            },
            "Dolomites": {
                "MR": 425,
                "MRSTD": 75,
                "Type": "Sedimentary",
                "mi": 9,
                "miSTD": 3,
            },
            "Gabbro": {
                "MR": 450,
                "MRSTD": 50,
                "Type": "Sedimentary",
                "mi": 27,
                "miSTD": 3,
            },
            "Gneiss": {
                "MR": 525,
                "MRSTD": 225,
                "Type": "Metamorphic",
                "mi": 28,
                "miSTD": 5,
            },
            "Granite": {
                "MR": 425,
                "MRSTD": 125,
                "Type": "Igneous",
                "mi": 32,
                "miSTD": 3,
            },
            "Granodiorite": {
                "MR": 425,
                "MRSTD": 125,
                "Type": "Igneous",
                "mi": 29,
                "miSTD": 3,
            },
            "Greywackes": {
                "MR": 350,
                "MRSTD": 0,
                "Type": "Sedimentary",
                "mi": 18,
                "miSTD": 3,
            },
            "Gypsum": {
                "MR": 350,
                "MRSTD": 0,
                "Type": "Sedimentary",
                "mi": 8,
                "miSTD": 2,
            },
            "Hornfels": {
                "MR": 550,
                "MRSTD": 150,
                "Type": "Metamorphic",
                "mi": 19,
                "miSTD": 4,
            },
            "Limestone (Crystalline)": {
                "MR": 500,
                "MRSTD": 100,
                "Type": "Sedimentary",
                "mi": 12,
                "miSTD": 3,
            },
            "Limestone (Micritic)": {
                "MR": 900,
                "MRSTD": 100,
                "Type": "Sedimentary",
                "mi": 9,
                "miSTD": 2,
            },
            "Limestone (Sparitic)": {
                "MR": 700,
                "MRSTD": 100,
                "Type": "Sedimentary",
                "mi": 10,
                "miSTD": 2,
            },
            "Marble": {
                "MR": 850,
                "MRSTD": 150,
                "Type": "Metamorphic",
                "mi": 9,
                "miSTD": 3,
            },
            "Marls": {
                "MR": 175,
                "MRSTD": 25,
                "Type": "Sedimentary",
                "mi": 7,
                "miSTD": 2,
            },
            "Metasandstones": {
                "MR": 250,
                "MRSTD": 50,
                "Type": "Metamorphic",
                "mi": 19,
                "miSTD": 3,
            },
            "Migmatite": {
                "MR": 375,
                "MRSTD": 25,
                "Type": "Metamorphic",
                "mi": 29,
                "miSTD": 3,
            },
            "Norite": {
                "MR": 375,
                "MRSTD": 25,
                "Type": "Igneous",
                "mi": 20,
                "miSTD": 5,
            },
            "Peridotite": {
                "MR": 275,
                "MRSTD": 25,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 5,
            },
            "Phyllites": {
                "MR": 550,
                "MRSTD": 250,
                "Type": "Metamorphic",
                "mi": 7,
                "miSTD": 3,
            },
            "Porphyries": {
                "MR": 400,
                "MRSTD": 0,
                "Type": "Igneous",
                "mi": 20,
                "miSTD": 5,
            },
            "Quartzites": {
                "MR": 375,
                "MRSTD": 75,
                "Type": "Metamorphic",
                "mi": 20,
                "miSTD": 3,
            },
            "Rhyolite": {
                "MR": 400,
                "MRSTD": 100,
                "Type": "Igneous",
                "mi": 25,
                "miSTD": 5,
            },
            "Sandstones": {
                "MR": 275,
                "MRSTD": 75,
                "Type": "Sedimentary",
                "mi": 17,
                "miSTD": 4,
            },
            "Schists": {
                "MR": 675,
                "MRSTD": 425,
                "Type": "Metamorphic",
                "mi": 12,
                "miSTD": 3,
            },
            "Shales": {
                "MR": 200,
                "MRSTD": 50,
                "Type": "Sedimentary",
                "mi": 6,
                "miSTD": 2,
            },
            "Siltstones": {
                "MR": 375,
                "MRSTD": 25,
                "Type": "Sedimentary",
                "mi": 7,
                "miSTD": 2,
            },
            "Slates": {
                "MR": 500,
                "MRSTD": 100,
                "Type": "Metamorphic",
                "mi": 7,
                "miSTD": 4,
            },
            "Tuff": {
                "MR": 300,
                "MRSTD": 100,
                "Type": "Igneous",
                "mi": 13,
                "miSTD": 5,
            },
        }
        with st.expander("Database for Different Rock Types"):
            RockType = st.selectbox(
                "Select the rock type. (Just write the letters to search.)",
                list(RockDict.keys()),
            )
            if (
                E_Method == "Generalized Hoek & Diederichs (2006)"
                or E_Method == "Yang (2006) / AASHTO"
            ):
                st.markdown(
                    "Modulus Ratio (MR):"
                    + str(RockDict[RockType]["MR"])
                    + " ± "
                    + str(RockDict[RockType]["MRSTD"])
                )
            st.markdown(
                "Material Constant for Intact Rock (mi):"
                + str(RockDict[RockType]["mi"])
                + " ± "
                + str(RockDict[RockType]["miSTD"])
            )

    with st.expander("Design aids"):
        st.markdown("*Click on the full size button at upper right corners to expand!*")

        col_im = st.columns(3)
        with col_im[0]:
            GSI_im = Image.open("images/GSI.png")
            st.image(GSI_im, caption="GSI Selection Chart")
        with col_im[1]:
            mi_im = Image.open("images/mi.png")
            st.image(mi_im, caption="Mi Values from Literature")
        with col_im[2]:
            D_im = Image.open("images/D.png")
            st.image(D_im, caption="Disturbance Factors for Various Applications")

    Rock = HoekBrown(UCS, GSI, mi, D, E_Method, sigma3_method, UW, H, MR, Name, Ei)

    with st.expander("ICE"):
        st.markdown(
            "Index of Elastic Behaviour as described by Celeda & Bieniawski - Ground Characterization and Structural Analyses for Tunnel Design - 2020"
        )
        D_tunnel = st.number_input("Tunnel diameter (m)", value=6.0)
        RMR = st.number_input("Rock Mass Rating (RMR)", value=GSI)
        K0 = st.number_input("Coefficient of lateral earth pressure", value=0.5)
        if H == None:
            H_tunnel = st.number_input("Tunnel depth (m)", value=30.0)
        else:
            H_tunnel = H
        F = 1.3 - 0.55 * (D_tunnel - 6) / 8
        if K0 < 1:
            ICE = (3704 * UCS * exp((RMR - 100) / 24) * F) / (3 - K0) / H_tunnel
        else:
            ICE = (3704 * UCS * exp((RMR - 100) / 24) * F) / (3 * K0 - 1) / H_tunnel
        ICE = round(ICE)
        st.markdown("**Index of Elastic Behaviour, ICE: **" + str(ICE))
        st.markdown("**Stress-Strain Behaviour:**")
        if ICE >= 70:
            if ICE > 130:
                SSB = "Fully elastic"
            else:
                SSB = "Near elastic"
            st.success("Stress-Strain Behaviour: " + SSB)
        elif ICE >= 40:
            SSB = "Moderate yielding"
            st.info("Stress-Strain Behaviour: " + SSB)
        else:
            if ICE >= 15:
                SSB = "Intense yielding"
            else:
                SSB = "Very intense yielding"
            st.error(SSB)

        st.markdown("**Full face excavation:**")
        if ICE >= 70:
            st.success("Full face method can be applied since ICE > 70.")
        else:
            st.error("Full face method CANNOT be applied since ICE < 70.")

        st.markdown("**Steel arches:**")
        if ICE >= 70:
            st.success("Steel arches are not needed since ICE > 70.")
        else:
            st.error("Steel arches are needed since ICE < 70.")

        st.markdown("**Borehole stability for bolts:**")
        if ICE >= 15:
            st.success("Boreholes can be drilled with sufficient borehole stability.")
        else:
            st.error("Borehole stability cannot be ensured, collapse may occur.")

    with cols[2]:
        st.write(Rock.fig)

    with cols[2]:
        add_selection = st.selectbox(
            "Additional Parameters", list(Rock.add_parameters.keys())
        )
        st.info(Rock.add_parameters[add_selection])


streamlitHoekBrown()
