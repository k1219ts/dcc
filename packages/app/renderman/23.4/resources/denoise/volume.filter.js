{
    "//": [
        "This file defines the parameters of the noise filter, specialized for volumes",
        "rendered in their own render pass (i.e. against transparent black).",
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

    "//": "Global settings which apply across all filteres/kernels:",
    "splitSpecularDiffuse": false,            "//": "If true, filters specular and diffuse separately",
    "splitSpecularDiffuseVariances": false,  "//": "When splitSpecularDiffuse and the file has diffuse_var and",
                                             "//": "specular_var layers, use them.",
    "warpType": "linear",                    "//": "cross-frame warp method: linear or nearest",
    
    "//": "'kernels' define individual bilateral, crossbilateral, or NLM filter kernels.  These ",
    "//": "are used when referenced in the 'filterbanks' section.",
    
    "kernels": [
        {
            "kernelname": "filter0_kernel_0",
            "kerneltype": "NLM",           "//": "type is NLM, crossBilateral, bilateral, or box.",
            "params": {
                "radius" : 1,              "//": "Spatial filter radius (pixels)",
                "sigma_s" : -1,            "//": "Spatial fall-off, negative for unweighted, full spatial range",

                "//": "The remaining settings must be the same for all kernels in a given 'filters.___' group.",
                
                "patch_radius" : 3,        "//": "Spatial filter patch radius (pixels)",
                
                "//": "Sigmas are tolerance values determine how much deviation between two pixels",
                "//": "is allowed when blending those pixels together in the bilateral filter.",
                "//": "0 means blend only when that feature is identical; large means the feature has",
                "//": "relatively little effect; -1 totally disables the feature.",

                "sigma_r" : 0.707106781,    "//": "unfiltered RGB difference tolerance",
                "sigma_alpha" : 0.7,        "//": "low-alpha tolerance; rec. > 0.2 (looser for this kernel since it allows_hair, which can have very low alpha)",
                
                "//": "disable depth, albedo, normal features since we don't typically have those features for volumes",
                "sigma_depth" : -1.0,        "//": "Z depth variance-normalized difference tolerance",
                "sigma_albedo" : -1.0,      "//": "albedo difference tolerance",
                "sigma_normal" : -1.0,      "//": "surface-normal variance-normalized tolerance",
                "allows_hair" : true,       "//": "true for smallest-radius filter, which will be used for hair to avoid over-blurring"
            }
        },
        {
            "kernelname": "filter1_kernel_0",
            "kerneltype": "NLM",
            "params": {
                "radius": 4,
                "sigma_s": -1,

                "//": "The remaining settings must be the same for all kernels in a given 'filters.___' group.",

                "patch_radius" : 3,
                "sigma_r": 0.707106781,
                "sigma_alpha" : 0.23,        "//": "low-alpha tolerance; rec. > 0.2 ",
                "sigma_depth": -1.0,
                "sigma_albedo": -1.0,
                "sigma_normal": -1.0,
                "allows_hair": false
            }
        },
        {
            "kernelname": "filter1_kernel_1",
            "kerneltype": "NLM",
            "params": {
                "radius": 7,
                "sigma_s": -1
            }
        },
        {
            "kernelname": "filter1_kernel_2",
            "kerneltype": "NLM",
            "params": {
                "radius": 12,
                "sigma_s": -1
            }
        }
    ],
    
    "filterbanks": {
        "//": "Diffuse and specular filterbanks to use if 'splitSpecularDiffuse' is true",
        "diffuse": {
            "strength": 1.0,                        "//": "A scale factor on color variance.",

            "sigmaAlphaForPrefilter": -1.0,         "//": "If > 0, sigma_alpha used when prefiltering variance.",
            "unpremultiplyColor": false,            "//": "If true, divides color by alpha.  Forced false when",
                                                    "//": "divideByAlbedo && !unpremultiplyFeatures since dividing",
                                                    "//": "color by albedo then implicitly divides by alpha", 
            "unpremultiplyFeatures": false,         "//": "If true, divides features (other than color) by alpha",
            "divideByAlbedo": false,                "//": "we normally divide by albedo to filter in a space where",
                                                    "//": "surface color textures have been removed.  You can set this",
                                                    "//": "false when albedo is causing trouble, but then you should also",
                                                    "//": "set unpremultiplyColor true.",
            "fireflyThreshold": 100.0,              "//": "determines how much higher a pixel's variance must be",
                                                    "//": "than its neighbors' average variance to be declared a firefly.",
                        
            "filters": [
                {
                    "filtername": "bilat_0",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter0_kernel_0"
                        ]
                    }
                },
                {
                    "filtername": "bilat_1",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter1_kernel_0",
                            "filter1_kernel_1",
                            "filter1_kernel_2"
                        ]
                    }
                }
            ]
        },
        "specular": {
            "strength": 1.0,                        "//": "A scale factor on color variance.",
            "sigmaAlphaForPrefilter": -1.0,         "//": "If > 0, sigma_alpha used when prefiltering variance.",
            "unpremultiplyColor": false,             "//": "If true, divides color by alpha.  Forced false when",
                                                    "//": "divideByAlbedo && !unpremultiplyFeatures since dividing",
                                                    "//": "color by albedo then implicitly divides by alpha", 
            "unpremultiplyFeatures": false,         "//": "If true, divides features (other than color) by alpha",
            "divideByAlbedo": false,
            "fireflyThreshold": 100.0,

            "filters": [
                {
                    "filtername": "bilat_0",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter0_kernel_0"
                        ]
                    }
                },
                {
                    "filtername": "bilat_1",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter1_kernel_0",
                            "filter1_kernel_1",
                            "filter1_kernel_2"
                        ]
                    }
                }
            ]
        },

        "//": "Filterbank to use if 'splitSpecularDiffuse' is false",
        "specularAndDiffuse": { 
            "strength": 1.0,                        "//": "A scale factor on color variance.",
            "sigmaAlphaForPrefilter": -1.0,         "//": "If > 0, sigma_alpha used when prefiltering variance.",
            "unpremultiplyColor": false,            "//": "If true, divides color by alpha.  Forced false when",
                                                    "//": "divideByAlbedo && !unpremultiplyFeatures since dividing",
                                                    "//": "color by albedo then implicitly divides by alpha", 
            "unpremultiplyFeatures": false,         "//": "If true, divides features (other than color) by alpha",
            "divideByAlbedo": false,
            "fireflyThreshold": 100.0,

            "filters": [
                {
                    "filtername": "bilat_0",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter0_kernel_0"
                        ]
                    }
                },
                {
                    "filtername": "bilat_1",
                    "params": {
                        "fireflyKiller": true,
                        "kernels": [
                            "filter1_kernel_0",
                            "filter1_kernel_1",
                            "filter1_kernel_2"
                        ]
                    }
                }
            ]
        }
    },

    "//": "'layergroups' defines groups of layers that can be filtered",
    "//": "together.  These may optionally include a single star that",
    "//": "will be pattern matched.  For example, if given a main",
    "//": "_variance.exr file along with a light17.exr containing",
    "//": "channels diffuse17.r, diffuse17.g, diffuse17.b, specular17.r,",
    "//": "specular17.g, and specular17.b, these will all be filtered",
    "//": "together as a group and written to light_filtered17.exr.",

    "layergroups": [
        {
            "emission": "emission*",    "//": "Optional, updated to sum of filtered diffuse and specular if both are present",
            "diffuse": "diffuse*",      "//": "Must have this, specular, or both",
            "specular": "specular*",    "//": "Must have diffuse, this, or both",
            "albedo": "albedo*",        "//": "Optional, name of albedo channels to use instead of variance image's albedo",
            "diffuseBank": "diffuse",   "//": "Optional, name of entry in filterbank block to use on diffuse layer",
            "specularBank": "specular", "//": "Optional, name of entry in filterbank block to use on specular layer"
        },
        {
            "emission": "directemission*",
            "diffuse": "directdiffuse*",
            "specular": "directspecular*",
            "albedo": "directalbedo*"
        },
        {
            "emission": "directEmission*",
            "diffuse": "directDiffuse*",
            "specular": "directSpecular*",
            "albedo": "directAlbedo*"
        },
        {
            "emission": "indirectemission*",
            "diffuse": "indirectdiffuse*",
            "specular": "indirectspecular*",
            "albedo": "indirectalbedo*"
        },
        {
            "emission": "indirectEmission*",
            "diffuse": "indirectDiffuse*",
            "specular": "indirectSpecular*",
            "albedo": "indirectAlbedo*"
        },
        {
            "diffuse": "emissive*",
            "albedo": "emissiveAlbedo*"
        },
        {
            "diffuse": "subsurface*",
            "albedo": "subsurfaceAlbedo*"
        },
        {
            "specular": "transmissive*",
            "albedo": "transmissiveAlbedo*"
        }
    ]
}
