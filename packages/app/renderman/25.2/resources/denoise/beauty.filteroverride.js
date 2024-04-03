{
    "//": [
        "This file defines the parameters of the noise filter.",
        "The two sections are:",
        "  'kernels': defines type type and parameters of a single filter kernel",
        "  'filters': groups one to four related kernels which are computed as a group.",
        "             Kernels in a given filter take all their parameters except",
        "             sigma_s from the first one in the group, and thus share most",
        "             parameter values.  Sigma_s should be different between kernels ",
        "             in a group. ",
        "",
        "Tip: to set sigma_s values equally-spaced in log space, use (eg in gnuplot):",
        "  sigma_s[i] = exp(log(sigma_s[3] / sigma_s[0]) / 3) ** i * sigma_s[0]",
        "  eg 0.33, 0.87, 2.29, 6.00"
    ],

    "splitSpecularDiffuse": false,
    "splitSpecularDiffuseVariances": false,

    "layergroups": [
        {
            "emission": "beauty*"
        }
    ]
}
