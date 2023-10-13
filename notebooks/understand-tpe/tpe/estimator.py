import numpy as np
import scipy
import matplotlib.pyplot as plt


class TPE:
    """
    This is a very simple implementation of the TPE algorithm just for learning porpuses.
    The Tree-Structured Parzen Estimator (TPE) is a Bayesian optimization algorithm that builds a
    probability density function (PDF) of the objective function using a tree structure.
    The algorithm searches for the maximum of the PDF, which is an estimation of the optimal solution.
    """

    def __init__(self, gamma=0.33):
        """
        gamma : a hyperparameter of the TPE algorithm that determines the fraction of the samples that are considered "low" (i.e., below the gamma-quantile of the values in Dy).
        Dx : a list of the sampled values of the input variable x.
        Dy : a list of the corresponding values of the objective function f(x).
        list_of_tries : a list that stores the results of the TPE algorithm at each iteration.
        function : the objective function to optimize.
        """
        self.Dx = []
        self.Dy = []
        self.gamma = gamma
        self.list_of_tries = []
        self.function = None

    def divide(self):
        """
        The divide method is used to split the dataset into two subsets based on the gamma-quantile of the values in Dy.
        The method returns two arrays, one containing the values of x corresponding to the "low" values of y,
        and another containing the values of x corresponding to the "high" values of y.
        """
        x = self.Dx
        y = self.Dy
        yp = np.quantile(y, self.gamma)
        l_index = [i for i, val in enumerate(y) if val < yp]
        g_index = [i for i, val in enumerate(y) if val >= yp]
        return x[l_index], x[g_index], yp

    def density(self, x, a, b, D):
        """
        x: a value at which to evaluate the PDF
        a: the lower bound of the truncated distribution
        b: the upper bound of the truncated distribution
        D: a list of data points used to estimate the distribution

        This is a method that calculates the probability density function (PDF)
        of a truncated normal distribution, given a set of data points (D) and the lower
        and upper bounds of the truncated distribution (a and b).
        """
        D.sort()
        pdf = 0
        # first iteration
        xL = a
        if len(D) == 1:
            xR = b
        else:
            xR = D[1]
        epsilon = (b - a) / min(100, 1 + len(D))
        sigma = min(max(D[0] - xL, xR - D[0], epsilon), b - a)
        a_prima, b_prima = (a - D[0]) / sigma, (b - D[0]) / sigma
        pdf += scipy.stats.truncnorm.pdf(x, a_prima, b_prima, loc=D[0], scale=sigma)
        if len(D) == 1:
            return pdf
        for i, xi in enumerate(D[1:-1]):
            xL = D[i - 1]
            xR = D[i + 1]
            sigma = min(max(xi - xL, xR - xi, epsilon), b - a)
            a_prima, b_prima = (a - xi) / sigma, (b - xi) / sigma
            pdf += scipy.stats.truncnorm.pdf(x, a_prima, b_prima, loc=xi, scale=sigma)
        # last iteration
        xL = D[-2]
        xR = b
        sigma = min(max(D[-1] - xL, xR - D[-1], epsilon), b - a)
        a_prima, b_prima = (a - D[-1]) / sigma, (b - D[-1]) / sigma
        pdf += scipy.stats.truncnorm.pdf(x, a_prima, b_prima, loc=D[-1], scale=sigma)
        return pdf / len(D)

    def optimize(self, a, b, f=None, n=30, n0=5, ns=20, n_equis=500):
        """
        a : start of search space.
        b : end of search space.
        f : function to optimize. When None, it use the saved function (if there is any).
        n : number of iterations after initial sampling.
        n0 : initial sampling size.
        ns : sampling size to maximize Ei(xi) or l(xi)/g(xi).
        n_equis: sampling size for plotting probability densities l(x) and g(x).
        """
        if not f:
            f = self.function
            if not f:
                print("Error: You Must Provide a Function to optimize.")
                return -1
        else:
            self.function = f

        x_equispaced = np.linspace(a - (b - a) / 100, b + (b - a) / 100, n_equis)
        # x_equispaced = np.linspace(a, b, n_equis)
        # initial sampling
        if len(self.Dx) == 0:
            self.Dx = np.random.uniform(a, b, n0)
            self.Dy = f(self.Dx)
            Dl, Dg, yp = self.divide()
            l = lambda x: self.density(x, a, b, Dl)
            g = lambda x: self.density(x, a, b, Dg)
            self._log_try(
                xi=self.Dx[-1],
                yi=self.Dy[-1],
                yp=yp,
                x_density=x_equispaced,
                l_density=l(x_equispaced),
                g_density=g(x_equispaced),
                Dl=Dl,
                Dg=Dg,
            )
        else:
            Dl, Dg, yp = self.divide()
            l = lambda x: self.density(x, a, b, Dl)
            g = lambda x: self.density(x, a, b, Dg)

        for i in range(n):
            xs = np.random.uniform(a, b, ns)
            xp = xs[np.argmax(l(xs) / g(xs))]
            # evaluar funcion
            yp = f(xp)
            self.Dx = np.append(self.Dx, xp)
            self.Dy = np.append(self.Dy, yp)
            Dl, Dg, yp = self.divide()
            l = lambda x: self.density(x, a, b, Dl)
            g = lambda x: self.density(x, a, b, Dg)
            ####
            # plot(a, b, l, g, f, yp, Dl=Dl, Dg=Dg)
            ####
            self._log_try(
                xi=self.Dx[-1],
                yi=self.Dy[-1],
                yp=yp,
                x_density=x_equispaced,
                l_density=l(x_equispaced),
                g_density=g(x_equispaced),
                Dl=Dl,
                Dg=Dg,
            )

    def _log_try(self, xi, yi, yp, x_density, l_density, g_density, Dl, Dg):
        """
        Add dictionary of try to the list of tries
        """
        self.list_of_tries.append(
            {
                "xi": xi,
                "yi": yi,
                "yp": yp,
                "x_density": x_density,
                "l_density": l_density,
                "g_density": g_density,
                "Dl": Dl,
                "Dg": Dg,
            }
        )

    def best_try(self):
        """
        return the best try as a tuple (best_x, best_y)
        """
        best_y = np.min(self.Dy)
        best_x = self.Dx[np.argmin(self.Dy)]
        return (best_x, best_y)

    def plot_try(
        self,
        try_index=-1,
        show_x_tries=True,
        show_EI=True,
        save=None,
        filename="fig",
        show=True,
    ):
        """
        plot the density functions of try indicated

        try_index : the index of the try you want to plot.
        show_x_tries : show tries.
        show_EI: show adquisition function EI(x) = l(x)/g(x)
        save : save plot into a file.
        filename : name of the file to save.
        show : use plt.show() or not
        """
        if not self.list_of_tries:
            print("Error: You neet to optimize first!")
            return -1

        try_dict = self.list_of_tries[try_index]
        x = try_dict["x_density"]
        yp = try_dict["yp"]
        Dl = try_dict["Dl"]
        Dg = try_dict["Dg"]
        l_density = try_dict["l_density"]
        g_density = try_dict["g_density"]
        f = self.function

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.scatter(Dl, f(Dl), c="blue")
        ax1.scatter(Dg, f(Dg), c="orange")
        if show_x_tries:
            ax1.scatter(
                Dl, np.zeros_like(Dl), c="blue", marker="x", label="Intentos buenos"
            )  # "good tries")
            ax1.scatter(
                Dg, np.zeros_like(Dg), c="orange", marker="x", label="Intentos malos"
            )  # "bad tries")
        ax1.plot(
            x, f(x), "--", c="red", label="Función Objetiva"
        )  # "Objective Function")
        ax1.set_ylabel(r"$f(x)$", c="red")
        ax1.plot(x, yp * np.ones_like(x), "--", c="green", label=r"$y*$")
        delta_y = ax1.get_ylim()[1] - ax2.get_ylim()[0]
        ax1.annotate(r"$y*$", (x[-1], yp), (x[-1], yp + delta_y / 20), c="green")
        ax1.legend()

        ax2.plot(x, l_density / np.sum(l_density), label=r"$l(x)$")
        ax2.plot(x, g_density / np.sum(g_density), label=r"$g(x)$")
        if show_EI:
            EI = l_density / g_density / len(x) / 1.1
            ax2.plot(x, EI, "--", label=r"EI(x)")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$p(x)$")
        ax2.legend()
        if save:
            plt.savefig(f"{filename}.png", dpi=150)
        if show:
            plt.show()

    def plot_evolution(self):
        iterations = np.arange(1, self.Dy.size + 1)
        mins = [self.Dy[0]]
        for y in self.Dy[1:]:
            mins.append(min(y, mins[-1]))
        plt.scatter(iterations, self.Dy)
        plt.plot(iterations, mins, "--", color="red", label="min value found")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()
