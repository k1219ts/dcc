[ Model ]
asset / $ASSET / model / model.payload.usd
                       / model.usd
                       / $VER / $ASSET.payload.usd
                              / $ASSET.usd
                              / $NODE.$TYPE_attr.usd
                              / $NODE.$TYPE_geom.usd
               / element / element.payload.usd
                         / element.usd
                         / $EL / $EL.payload.usd
                               / $EL.usd
                               / model / model.usd
                                       / $VER / $EL.payload.usd
                                              / $EL.usd
                                              / $NODE.$TYPE_attr.usd
                                              / $NODE.$TYPE_geom.usd

[ Model - Clip ]
asset / $ASSET / clip / clip.payload.usd
                      / clip.usd
                      / $VER / loopClip.payload.usd
                             / loopClip.usd
                             / $NS_clip / $ASSET.usd
                                        / $NODE.$TYPE_attr.usd
                                        / $NODE.$TYPE_geom.usd
                             / $NS_loop$SCALE / $NS_loop$SCALE.payload.usd
                                              / $NS_loop$SCALE.usd
                                              / $NODE.$TYPE_geom_loop.usd

[ Texture ]
asset / $ASSET / texture / texture.usd
                         / tex / tex.usd
                               / $VER / tex.payload.usd
                                      / tex.version.usd

asset / $ASSET / element / $EL / texture / texture.usd
                                         / tex / tex.usd
                                               / $VER / tex.payload.usd
                                                      / tex.version.usd

asset / $ASSET / texture / $REFASSET / texture.override.usd
                                   / texture.usd
                                   / tex / tex.usd
                                         / $VER / tex.payload.usd
                                                / tex.version.usd

[ RIG - ASSET ]
asset / $ASSET / rig / rig.payload.usd
                     / rig.usd
                     / usd / $RIG_VER / $NODE.payload.usd
                                      / $NODE.$TYPE_attr.usd
                                      / $NODE.$TYPE_geom.usd
[ RIG - SHOT ]
shot / $SEQ /$SHOT / ani / ani.payload.usd
                         / ani.usd
                         / $NS / $NS.payload.usd
                               / $NS.usd
                               / $VER / $NS.payload.usd
                                      / $NS.usd
                                      / $NODE.xform.usd
                                      / $NODE.$TYPE_attr.usd
                                      / $NODE.$TYPE_geom.usd

[ Crowd ]
asset / $ASSET / agent / agent.payload.usd
                       / agent.usd
                       / $ASSET_$TYPE / $ASSET_$TYPE.payload.usd
                                      / $ASSET_$TYPE.usd
                                      / $VER / $ASSET_$TYPE.payload.usd
                                             / $ASSET_$TYPE.usd
                                             / $NODE.attr.usd
                                             / $NODE.geom.usd

[ Zenn ]
asset / $ASSET / zenn / zenn.payload.usd
                      / zenn.usd
                      / $VER / zn_deforms.payload.usd
                             / zn_deforms.usd
                             / zn_deforms.$TYPE_geom.usd
                             / $ZN_Deform_NODE / $ZN_Deform_NODE.$TYPE_geom.usd
                             / $ZN_Deform_NODE / $ZN_Deform_NODE.$TYPE_geom.usd
