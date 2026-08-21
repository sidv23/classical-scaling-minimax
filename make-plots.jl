import Pkg;
Pkg.activate(@__DIR__);

begin
    using JLD2
    using Random, Statistics, VectorizedStatistics
    using Plots, Plots.Measures, LaTeXStrings
end

for filename in readdir("src")
    if endswith(filename, ".jl")
        includet(joinpath("src", filename))
    end
end

begin
    data = jldopen("results/simulations.jld2")
    sim_res = data["sim_res"]
    Ns = data["Ns"]
    Qs = data["Qs"]
    Sigmas = data["Sigmas"]
    Ks = data["Ks"]
    Noises = data["Noises"]
end


begin
    agg(x, f, dims=5) = dropdims(f(x, dims=dims), dims=dims)
    
    dummy_plot(label, cls, text=x -> "$x", title="", after="") = plot(
        zeros(1, 3);
        ms=3, lw=2, la=1, a=0,
        label=map(x -> text(x) * after, label) |> permutedims,
        legend=:top,
        showaxis=false, grid=false, legend_columns=-1,
        foreground_color_legend=nothing,
        c=cls, legendfontsize=12, 
        bottom_margin=0 * Plots.mm,
        top_margin=-1 * Plots.mm,
        title=title,
    )
    quant(alpha) = (x; kwargs...) -> vquantile(x, alpha; kwargs...)
end

############
begin
    dm_1() = dummy_plot(
        Sigmas, cls3,
        x -> L"\sigma = {%$x}",
        L"L${}_{\mathrm{rmse}}$",
        L"\ \ \ \ \ \ \ \ \ \ \ "
    )
    l_1()  = @layout [
        a{0.01h}
        grid(length(Ks), length(Qs))
    ]
end

#######
begin
    cls3 = [:firebrick1 :dodgerblue1 :chartreuse2]
    res = @pipe sim_res[:additive] |> map(x -> x[2], _)
    # Ns, Qs, Ks, Sigmas
    Mid = agg(res, median)
    Q1 = agg(res, quant(0.9))
    Q2 = agg(res, quant(0.1))
    plts = [
        begin
            m = Mid[:, i, j, :]
            q1 = Q1[:, i, j, :]
            q2 = Q2[:, i, j, :]
            p = plot(Ns,
                m,
                ribbon=(
                    m .- q1,
                    q2 .- m
                ),
                fa=0.2, m=:o, msw=0, lw=1.5,
                label="",
                c=cls3,
                title=j == 1 ? L"q=%$(round(Int, q))" : "",
                ylabel=i == 1 ? L"\kappa=%$kappa" : "",
                xlabel=j == length(Ks) ? L"n" : "",
                xticks=j == length(Ks) ? true : false,
                yticks=i == 1 ? true : false,
                top_margin=j == 1 ? -5 * Plots.mm : 0 * Plots.mm,
                bottom_margin=j == length(Ks) ? 0 * Plots.mm : -3 * Plots.mm,
                left_margin=i == 1 ? 7 * Plots.mm : -3 * Plots.mm,
                right_margin=i == length(Qs) ? 2 * Plots.mm : 1 * Plots.mm,
                grid=false
            )
            for (m, s) in enumerate(Sigmas)
                p = plot(p, Ns, n -> 3.0 * sqrt(q/(q-2)) * s * kappa .* √(1 / n), ls=:dash, label="", c=cls3[m])
            end
            p
        end
        for (i, q) in enumerate(Qs), (j, kappa) in enumerate(Ks)
    ]
    dm = dm_1()
    dm = title!(dm, "\n" * L"L${}_{\textrm{rmse}}$")
    dm = plot!(dm, zeros(1), c=:black, ls=:dash, label=L"${\sigma\kappa} \cdot \sqrt{1/n}$")
    p11 = plot(dm, plts..., layout=l_1(), size=(900, 1000), scale=:log10, ylim=(2e-3, 5e-1))
end
savefig(p11, "plots/p11.pdf")


#######
begin
    cls3 = [:firebrick1 :dodgerblue1 :chartreuse2]
    res = @pipe sim_res[:additive] |> map(x -> x[1], _)
    Mid = agg(res, median)
    Q1 = agg(res, quant(0.9))
    Q2 = agg(res, quant(0.1))
    plts = [
        begin
            m = Mid[:, i, j, :]
            q1 = Q1[:, i, j, :]
            q2 = Q2[:, i, j, :]
            p = plot(Ns,
                m,
                ribbon=(
                    m .- q1,
                    q2 .- m
                ),
                fa=0.2, m=:o, msw=0, lw=1.5,
                label="",
                c=cls3,
                title=j == 1 ? L"q=%$(round(Int, q))" : "",
                ylabel=i == 1 ? L"\kappa=%$kappa" : "",
                xlabel=j == length(Ks) ? L"n" : "",
                xticks=j == length(Ks) ? true : false,
                yticks=i == 1 ? true : false,
                top_margin=j == 1 ? -5 * Plots.mm : 0 * Plots.mm,
                bottom_margin=j == length(Ks) ? 0 * Plots.mm : -3 * Plots.mm,
                left_margin=i == 1 ? 7 * Plots.mm : -3 * Plots.mm,
                right_margin=i == length(Qs) ? 2 * Plots.mm : 1 * Plots.mm,
                grid=false
            )
            for (m, s) in enumerate(Sigmas)
                p = plot(p, Ns, n -> 2.0 * s * sqrt(q/(q-2)) * kappa .* √(log(n) / n), ls=:dash, label="", c=cls3[m])
            end
            p
        end
        for (i, q) in enumerate(Qs), (j, kappa) in enumerate(Ks)
    ]
    dm = dm_1()
    dm = title!(dm, "\n" * L"L${}_{2\!\!\to\!\!\infty}$")
    dm = plot!(dm, zeros(1), c=:black, ls=:dash, label=L"${\sigma\kappa^2} \cdot \sqrt{\log{n}/n}$")
    p12 = plot(dm, plts..., layout=l_1(), size=(900, 1000), scale=:log10, ylim=(5e-3, 1e1))
end
savefig(p12, "plots/p12.pdf")




############
begin
    dm_2() = dummy_plot(
        Qs, cls3_alt,
        x -> L"t_{%$(round(Int, x))}",
        L"L${}_{\mathrm{rmse}}$",
        L"\ \ \ \ \ \ \ \ \ \ \ "
        )
    l_2() = @layout [
        a{0.01h}
        grid(length(Noises), length(Sigmas))
    ]
end

####
begin
    cls3_alt = [:purple :red :orange]
    res = [
        map(x -> x[2], v)[:, :, 2, :, :]
        for (k, v) in sim_res
    ]
    plts = [
        begin
            m = agg(res[j], median, 4)[:, :, i]
            q1 = agg(res[j], quant(0.9), 4)[:, :, i]
            q2 = agg(res[j], quant(0.1), 4)[:, :, i]
            p = plot(Ns,
                m,
                ribbon=(
                    m .- q1,
                    q2 .- m
                ),
            fa = 0.2, m=:o, msw=0, lw=1.5,
            label="",
            c=cls3_alt,
            title=j == 1 ? L"\sigma=%$s" : "",
            ylabel=i == 1 ? "$noise" : "",
            xlabel=j == length(Noises) ? L"n" : "",
            xticks=j == length(Noises) ? true : false,
            yticks=i == 1 ? true : false,
            top_margin=j == 1 ? -5 * Plots.mm : 0 * Plots.mm,
            bottom_margin=j == length(Noises) ? 0 * Plots.mm : -3 * Plots.mm,
            left_margin=i == 1 ? 7 * Plots.mm : -3 * Plots.mm,
            right_margin=i == length(Sigmas) ? 2 * Plots.mm : 1 * Plots.mm,
            grid=false
        )
        p = plot(p, Ns, n -> 2.3 * s .* √(1/n), ls=:dash, label="", c=:black)
        p
    end
        for (i, s) in enumerate(Sigmas), (j, noise) in enumerate(Noises)
    ]
    dm = dm_2()
    dm = title!(dm, "\n" * L"L${}_{\mathrm{rmse}}$")
    dm = plot!(dm, zeros(1), c=:black, ls=:dash, label=L"${\sigma} \cdot \sqrt{1/n}$")
    p21 = plot(dm, plts..., layout=l_2(), size=(300, 200) .* (length(Noises), length(Sigmas)), scale=:log10, ylim=(2e-3, 1e-0))
end
savefig(p21, "plots/p21.pdf")




####
begin
    cls3_alt = [:purple :red :orange]
    res = [
        map(x -> x[1], v)[:, :, 2, :, :]
        for (k, v) in sim_res
    ]
    plts = [
        begin
            m = agg(res[j], median, 4)[:, :, i]
            q1 = agg(res[j], quant(0.9), 4)[:, :, i]
            q2 = agg(res[j], quant(0.1), 4)[:, :, i]
            p = plot(Ns,
                m,
                ribbon=(
                    m .- q1,
                    q2 .- m
                ),
                fa=0.2, m=:o, msw=0, lw=1.5,
                label="",
                c=cls3_alt,
                title=j == 1 ? L"\sigma=%$s" : "",
                ylabel=i == 1 ? "$noise" : "",
                xlabel=j == length(Noises) ? L"n" : "",
                xticks=j == length(Noises) ? true : false,
                yticks=i == 1 ? true : false,
                top_margin=j == 1 ? -5 * Plots.mm : 0 * Plots.mm,
                bottom_margin=j == length(Noises) ? 0 * Plots.mm : -3 * Plots.mm,
                left_margin=i == 1 ? 7 * Plots.mm : -3 * Plots.mm,
                right_margin=i == length(Sigmas) ? 2 * Plots.mm : 1 * Plots.mm,
                grid=false
            )
            p = plot(p, Ns, n -> 2 * s .* √(log(n) / n), ls=:dash, label="", c=:black)
            p
        end
        for (i, s) in enumerate(Sigmas), (j, noise) in enumerate(Noises)
    ]
    dm = dm_2()
    dm = title!(dm, "\n" * L"L${}_{2\!\!\to\!\!\infty}$")
    dm = plot!(dm, zeros(1), c=:black, ls=:dash, label=L"${\sigma} \cdot \sqrt{\log{n}/n}$")
    p22 = plot(dm, plts..., layout=l_2(), size=(300, 230) .* (length(Noises), length(Sigmas)), scale=:log10, ylim=(5e-3, 2e1))
end
savefig(p22, "plots/p22.pdf")
