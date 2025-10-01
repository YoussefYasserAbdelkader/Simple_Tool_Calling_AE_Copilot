from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict
from xsdata_pydantic.fields import field


class AeAnalyseSimtime(BaseModel):
    model_config = ConfigDict(defer_build=True)
    start_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "StartTime",
            "type": "Attribute",
        },
    )
    end_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "EndTime",
            "type": "Attribute",
        },
    )


class AeAxiBusType(BaseModel):
    """
    AXI bus configuration.

    :ivar width: AXI-BUS width in Bytes.
    :ivar frequency: AXI-BUS frequency in Hz.
    """

    model_config = ConfigDict(defer_build=True)
    width: int = field(
        default=4,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
        },
    )
    frequency: int = field(
        default=100000000,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1000000,
        },
    )


class AeCpuFrequency(BaseModel):
    """
    This specifies the CPU Frequency in Hz.
    """

    model_config = ConfigDict(defer_build=True)
    value: int = field(
        metadata={
            "type": "Attribute",
            "required": True,
            "min_inclusive": 1000,
            "max_inclusive": 1000000000,
        }
    )


class AeDualCoreClusterCoresPerCluster(Enum):
    VALUE_1 = 1
    VALUE_2 = 2


class AeEndpointDmaconfigType(BaseModel):
    class Meta:
        name = "AeEndpointDMAConfigType"

    model_config = ConfigDict(defer_build=True)
    frequency: int = field(
        metadata={
            "name": "Frequency",
            "type": "Attribute",
            "required": True,
            "min_inclusive": 1,
        }
    )


class AeEthernetInterfaceTypeMode(Enum):
    SIMULATED = "simulated"
    NATIVE = "native"


class AeGenericHwfrequency(BaseModel):
    """
    This specifies the Generic HardwareIP Frequency in Hz.
    """

    class Meta:
        name = "AeGenericHWFrequency"

    model_config = ConfigDict(defer_build=True)
    value: int = field(
        metadata={
            "type": "Attribute",
            "required": True,
            "min_inclusive": 1000,
            "max_inclusive": 1000000000,
        }
    )


class AeHexaCoreClusterCoresPerCluster(Enum):
    VALUE_1 = 1
    VALUE_2 = 2
    VALUE_4 = 4
    VALUE_8 = 8
    VALUE_16 = 16


class AeHwSwMappingType(BaseModel):
    """
    :ivar cluster_prebuilt_application_mapping:
    :ivar core_prebuilt_application_mapping:
    :ivar core_runnable_mapping:
    :ivar cluster_ref: Path to the required cluster, eg.: /ecu-name/soc-
        name/cluster-name or /ecu-name/soc-name/chiplet-name/cluster-
        name
    """

    model_config = ConfigDict(defer_build=True)
    cluster_prebuilt_application_mapping: list[
        "AeHwSwMappingType.ClusterPrebuiltApplicationMapping"
    ] = field(
        default_factory=list,
        metadata={
            "name": "Cluster-PrebuiltApplication-Mapping",
            "type": "Element",
        },
    )
    core_prebuilt_application_mapping: list[
        "AeHwSwMappingType.CorePrebuiltApplicationMapping"
    ] = field(
        default_factory=list,
        metadata={
            "name": "Core-PrebuiltApplication-Mapping",
            "type": "Element",
        },
    )
    core_runnable_mapping: list["AeHwSwMappingType.CoreRunnableMapping"] = (
        field(
            default_factory=list,
            metadata={
                "name": "Core-Runnable-Mapping",
                "type": "Element",
            },
        )
    )
    cluster_ref: str = field(
        metadata={
            "name": "ClusterRef",
            "type": "Attribute",
            "required": True,
        }
    )

    class ClusterPrebuiltApplicationMapping(BaseModel):
        """
        :ivar prebuilt_application_ref: Path to the required prebuilt
            application, eg.: /prebuilt-application-name
        """

        model_config = ConfigDict(defer_build=True)
        prebuilt_application_ref: str = field(
            metadata={
                "name": "PrebuiltApplicationRef",
                "type": "Attribute",
                "required": True,
            }
        )

    class CorePrebuiltApplicationMapping(BaseModel):
        """
        :ivar core_id: Required core Id starting from index 0, must be
            less than the CoresPerCluster value.
        :ivar prebuilt_application_ref: Path to the required prebuilt
            application, eg.: /prebuilt-application-name
        """

        model_config = ConfigDict(defer_build=True)
        core_id: int = field(
            metadata={
                "name": "CoreId",
                "type": "Attribute",
                "required": True,
                "min_inclusive": 0,
                "max_inclusive": 7,
            }
        )
        prebuilt_application_ref: str = field(
            metadata={
                "name": "PrebuiltApplicationRef",
                "type": "Attribute",
                "required": True,
            }
        )

    class CoreRunnableMapping(BaseModel):
        """
        :ivar core_id: Required core Id starting from index 0, must be
            less than the CoresPerCluster value.
        :ivar runnable_ref: Path to the required runnable, eg.: /swc-
            name/swc-internal-behavior-name/runnable-name
        :ivar load: The number of CPU cycles used as synthetic load
        :ivar priority:
        """

        model_config = ConfigDict(defer_build=True)
        core_id: int = field(
            metadata={
                "name": "CoreId",
                "type": "Attribute",
                "required": True,
                "min_inclusive": 0,
                "max_inclusive": 7,
            }
        )
        runnable_ref: str = field(
            metadata={
                "name": "RunnableRef",
                "type": "Attribute",
                "required": True,
            }
        )
        load: int = field(
            metadata={
                "name": "Load",
                "type": "Attribute",
                "required": True,
            }
        )
        priority: int = field(
            metadata={
                "name": "Priority",
                "type": "Attribute",
                "required": True,
                "min_inclusive": 1,
                "max_inclusive": 99,
            }
        )


class AeIdentity(BaseModel):
    model_config = ConfigDict(defer_build=True)
    name: str = field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class AeOctaCoreClusterCoresPerCluster(Enum):
    VALUE_1 = 1
    VALUE_2 = 2
    VALUE_4 = 4
    VALUE_8 = 8


class AeOsType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    nucleus_rtos: Optional["AeOsType.NucleusRtos"] = field(
        default=None,
        metadata={
            "name": "Nucleus_RTOS",
            "type": "Element",
        },
    )
    linux: Optional["AeOsType.Linux"] = field(
        default=None,
        metadata={
            "name": "Linux",
            "type": "Element",
        },
    )

    class NucleusRtos(BaseModel):
        """Siemens Nucleus® RTOS enables system developers to address the complex
        requirements demanded by today’s advanced embedded designs.

        Nucleus brings together kernel-rich functionality and tooling
        features ideal for applications where a scalable footprint,
        connectivity, security, power management, and deterministic
        performance are essential. Nucleus RTOS is a proven, reliable,
        and fully optimized RTOS. Nucleus has been successfully deployed
        in highly demanding markets with rigorous safety and security
        requirements such as industrial systems, medical devices,
        airborne systems, automotive and more.
        """

        model_config = ConfigDict(defer_build=True)
        affine_tasks_by_os: bool = field(
            default=False,
            metadata={
                "name": "Affine-tasks-byOS",
                "type": "Attribute",
            },
        )
        show_uart_terminal: bool = field(
            default=False,
            metadata={
                "name": "Show_UART_Terminal",
                "type": "Attribute",
            },
        )

    class Linux(BaseModel):
        """
        :ivar buildroot_file_system: Buildroot Simplified File System.
        :ivar ubuntu_file_system: Ubuntu File System.
        :ivar affine_tasks_by_os:
        :ivar show_uart_terminal:
        """

        model_config = ConfigDict(defer_build=True)
        buildroot_file_system: Optional[str] = field(
            default=None,
            metadata={
                "name": "Buildroot_File_System",
                "type": "Element",
            },
        )
        ubuntu_file_system: Optional[str] = field(
            default=None,
            metadata={
                "name": "Ubuntu_File_System",
                "type": "Element",
            },
        )
        affine_tasks_by_os: bool = field(
            default=False,
            metadata={
                "name": "Affine-tasks-byOS",
                "type": "Attribute",
            },
        )
        show_uart_terminal: bool = field(
            default=False,
            metadata={
                "name": "Show_UART_Terminal",
                "type": "Attribute",
            },
        )


class AePowerParameterType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    split_power_mw: float = field(
        default=10.0,
        metadata={
            "name": "Split_power_mw",
            "type": "Attribute",
            "min_inclusive": 0.0,
        },
    )
    delay_power_mw: float = field(
        default=10.0,
        metadata={
            "name": "Delay_power_mw",
            "type": "Attribute",
            "min_inclusive": 0.0,
        },
    )
    sequential_power_mw: float = field(
        default=12.0,
        metadata={
            "name": "Sequential_power_mw",
            "type": "Attribute",
            "min_inclusive": 0.0,
        },
    )
    static_power_leakage_mw: float = field(
        default=0.05,
        metadata={
            "name": "Static_Power_Leakage_mw",
            "type": "Attribute",
            "min_inclusive": 0.0,
        },
    )
    clock_tree_power_mw: float = field(
        default=1.0,
        metadata={
            "name": "Clock_Tree_Power_mw",
            "type": "Attribute",
            "min_inclusive": 0.0,
        },
    )
    power_per_nominal_clock_mhz: int = field(
        default=100,
        metadata={
            "name": "Power_Per_Nominal_Clock_Mhz",
            "type": "Attribute",
            "min_inclusive": 1,
        },
    )


class AeQuadCoreClusterCoresPerCluster(Enum):
    VALUE_1 = 1
    VALUE_2 = 2
    VALUE_4 = 4


class AeReference(BaseModel):
    model_config = ConfigDict(defer_build=True)
    dest: str = field(
        metadata={
            "name": "DEST",
            "type": "Attribute",
            "required": True,
        }
    )


class AeSingleCoreClusterCoresPerCluster(Enum):
    VALUE_1 = 1


class AeUcieInterfaceTypeMode(Enum):
    HOST = "host"
    ENDPOINT = "endpoint"


class ChipletLinkLatencyUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class LatencyUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class LogicalIfDelayUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class PeriodUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class PhysicalIfDelayUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class ProtocolConvLatencyUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class SimulationTimeUnit(Enum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"


class AeAnalysisType(BaseModel):
    """
    :ivar sw_analysis_enable:
    :ivar hw_analysis_enable:
    :ivar power_analysis_enable: Power Analysis can only be enabled for
        SoCs running Nucleus on all its CPU Clusters
    :ivar network_analysis_enable:
    """

    model_config = ConfigDict(defer_build=True)
    sw_analysis_enable: Optional["AeAnalysisType.SwAnalysisEnable"] = field(
        default=None,
        metadata={
            "name": "SW-Analysis-Enable",
            "type": "Element",
        },
    )
    hw_analysis_enable: Optional["AeAnalysisType.HwAnalysisEnable"] = field(
        default=None,
        metadata={
            "name": "HW-Analysis-Enable",
            "type": "Element",
        },
    )
    power_analysis_enable: Optional["AeAnalysisType.PowerAnalysisEnable"] = (
        field(
            default=None,
            metadata={
                "name": "Power-Analysis-Enable",
                "type": "Element",
            },
        )
    )
    network_analysis_enable: Optional[
        "AeAnalysisType.NetworkAnalysisEnable"
    ] = field(
        default=None,
        metadata={
            "name": "Network-Analysis-Enable",
            "type": "Element",
        },
    )

    class SwAnalysisEnable(BaseModel):
        model_config = ConfigDict(defer_build=True)
        all_so_cs: Optional[str] = field(
            default=None,
            metadata={
                "name": "ALL-SoCs",
                "type": "Element",
            },
        )
        selected_so_cs: Optional[
            "AeAnalysisType.SwAnalysisEnable.SelectedSoCs"
        ] = field(
            default=None,
            metadata={
                "name": "Selected-SoCs",
                "type": "Element",
            },
        )

        class SelectedSoCs(BaseModel):
            model_config = ConfigDict(defer_build=True)
            so_c: list[AeReference] = field(
                default_factory=list,
                metadata={
                    "name": "SoC",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )

    class HwAnalysisEnable(BaseModel):
        model_config = ConfigDict(defer_build=True)
        all_so_cs: Optional[str] = field(
            default=None,
            metadata={
                "name": "ALL-SoCs",
                "type": "Element",
            },
        )
        selected_so_cs: Optional[
            "AeAnalysisType.HwAnalysisEnable.SelectedSoCs"
        ] = field(
            default=None,
            metadata={
                "name": "Selected-SoCs",
                "type": "Element",
            },
        )

        class SelectedSoCs(BaseModel):
            model_config = ConfigDict(defer_build=True)
            so_c: list[AeReference] = field(
                default_factory=list,
                metadata={
                    "name": "SoC",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )

    class PowerAnalysisEnable(BaseModel):
        model_config = ConfigDict(defer_build=True)
        all_so_cs: Optional[str] = field(
            default=None,
            metadata={
                "name": "ALL-SoCs",
                "type": "Element",
            },
        )
        selected_so_cs: Optional[
            "AeAnalysisType.PowerAnalysisEnable.SelectedSoCs"
        ] = field(
            default=None,
            metadata={
                "name": "Selected-SoCs",
                "type": "Element",
            },
        )

        class SelectedSoCs(BaseModel):
            model_config = ConfigDict(defer_build=True)
            so_c: list[AeReference] = field(
                default_factory=list,
                metadata={
                    "name": "SoC",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )

    class NetworkAnalysisEnable(BaseModel):
        model_config = ConfigDict(defer_build=True)
        can_bus_monitor_enable: Optional[str] = field(
            default=None,
            metadata={
                "name": "CAN-BUS-Monitor-Enable",
                "type": "Element",
            },
        )
        eth_switch_monitor_enable: Optional[str] = field(
            default=None,
            metadata={
                "name": "ETH-Switch-Monitor-Enable",
                "type": "Element",
            },
        )


class AeCpuCluster(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    frequency: AeCpuFrequency = field(
        metadata={
            "name": "Frequency",
            "type": "Element",
            "required": True,
        }
    )
    cores_per_cluster: object = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeCpuCore(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    core_runnable_mapping: list["AeCpuCore.CoreRunnableMapping"] = field(
        default_factory=list,
        metadata={
            "name": "Core-runnable-Mapping",
            "type": "Element",
            "min_occurs": 1,
        },
    )

    class CoreRunnableMapping(BaseModel):
        """
        :ivar dest:
        :ivar load: The number of CPU cycles.
        :ivar priority:
        """

        model_config = ConfigDict(defer_build=True)
        dest: str = field(
            metadata={
                "name": "DEST",
                "type": "Attribute",
                "required": True,
            }
        )
        load: int = field(
            metadata={
                "name": "Load",
                "type": "Attribute",
                "required": True,
            }
        )
        priority: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )


class AeD2DconfigType(BaseModel):
    """
    :ivar link_width:
    :ivar link_frequency:
    :ivar protocol_conv_latency:
    :ivar chiplet_link_latency:
    :ivar logical_if_delay:
    :ivar physical_if_delay:
    :ivar dest_chiplet_ref: Path to the destination chiplet, eg.: /ecu-
        name/soc-name or /ecu-name/soc-name/chiplet-name
    """

    class Meta:
        name = "AeD2DConfigType"

    model_config = ConfigDict(defer_build=True)
    link_width: "AeD2DconfigType.LinkWidth" = field(
        metadata={
            "name": "LINK_WIDTH",
            "type": "Element",
            "required": True,
        }
    )
    link_frequency: "AeD2DconfigType.LinkFrequency" = field(
        metadata={
            "name": "LINK_FREQUENCY",
            "type": "Element",
            "required": True,
        }
    )
    protocol_conv_latency: "AeD2DconfigType.ProtocolConvLatency" = field(
        metadata={
            "name": "PROTOCOL_CONV_LATENCY",
            "type": "Element",
            "required": True,
        }
    )
    chiplet_link_latency: "AeD2DconfigType.ChipletLinkLatency" = field(
        metadata={
            "name": "CHIPLET_LINK_LATENCY",
            "type": "Element",
            "required": True,
        }
    )
    logical_if_delay: "AeD2DconfigType.LogicalIfDelay" = field(
        metadata={
            "name": "LOGICAL_IF_DELAY",
            "type": "Element",
            "required": True,
        }
    )
    physical_if_delay: "AeD2DconfigType.PhysicalIfDelay" = field(
        metadata={
            "name": "PHYSICAL_IF_DELAY",
            "type": "Element",
            "required": True,
        }
    )
    dest_chiplet_ref: str = field(
        metadata={
            "name": "DestChipletRef",
            "type": "Attribute",
            "required": True,
        }
    )

    class LinkWidth(BaseModel):
        """The width of the UCIe link (bytes per cycle).

        unit in Bytes
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=256,
            metadata={
                "type": "Attribute",
                "min_inclusive": 1,
            },
        )

    class LinkFrequency(BaseModel):
        """The operating frequency of the UCIe link.

        unit in Hz
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=250000000,
            metadata={
                "type": "Attribute",
                "min_inclusive": 1,
            },
        )

    class ProtocolConvLatency(BaseModel):
        """
        Time taken by the D2D adapter for protocol conversion.
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=6,
            metadata={
                "type": "Attribute",
            },
        )
        unit: ProtocolConvLatencyUnit = field(
            default=ProtocolConvLatencyUnit.NS,
            metadata={
                "type": "Attribute",
            },
        )

    class ChipletLinkLatency(BaseModel):
        """
        Time taken for data to traverse the UCIe link.
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=2,
            metadata={
                "type": "Attribute",
            },
        )
        unit: ChipletLinkLatencyUnit = field(
            default=ChipletLinkLatencyUnit.NS,
            metadata={
                "type": "Attribute",
            },
        )

    class LogicalIfDelay(BaseModel):
        """
        Delay introduced by data processing before transmission.
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=6,
            metadata={
                "type": "Attribute",
            },
        )
        unit: LogicalIfDelayUnit = field(
            default=LogicalIfDelayUnit.NS,
            metadata={
                "type": "Attribute",
            },
        )

    class PhysicalIfDelay(BaseModel):
        """
        Delay caused by signal propagation through physical media.
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            default=5,
            metadata={
                "type": "Attribute",
            },
        )
        unit: PhysicalIfDelayUnit = field(
            default=PhysicalIfDelayUnit.NS,
            metadata={
                "type": "Attribute",
            },
        )


class AeDataReceivedEventType(BaseModel):
    """
    :ivar short_name:
    :ivar start_on_event_ref: Example /SWC1/SWC1Behav/Runnable
    :ivar custom_behavior_ref: Example /Behavior1
    :ivar data_iref:
    """

    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    start_on_event_ref: AeReference = field(
        metadata={
            "name": "START-ON-EVENT-REF",
            "type": "Element",
            "required": True,
        }
    )
    custom_behavior_ref: Optional[AeReference] = field(
        default=None,
        metadata={
            "name": "CUSTOM-BEHAVIOR-REF",
            "type": "Element",
        },
    )
    data_iref: "AeDataReceivedEventType.DataIref" = field(
        metadata={
            "name": "DATA-IREF",
            "type": "Element",
            "required": True,
        }
    )

    class DataIref(BaseModel):
        """
        :ivar target_data_element_ref: Example
            /SWC2/sWC2_IntBehav/Runnable/DataElment
        """

        model_config = ConfigDict(defer_build=True)
        target_data_element_ref: list[AeReference] = field(
            default_factory=list,
            metadata={
                "name": "TARGET-DATA-ELEMENT-REF",
                "type": "Element",
                "min_occurs": 1,
            },
        )


class AeEthernetInterfaceType(BaseModel):
    """Ethernet mode:
    * Simulated: Default mode, not visible from the Host network
    * Native: Connected to the Host network through mapped ports
    ** In each SoC, only one chiplet can be in Native mode
    ** Chiplets in different ECUs must have same ethernet mode to communicate with each others"""

    model_config = ConfigDict(defer_build=True)
    mode: AeEthernetInterfaceTypeMode = field(
        default=AeEthernetInterfaceTypeMode.SIMULATED,
        metadata={
            "name": "Mode",
            "type": "Attribute",
        },
    )


class AeGenericHwdataReceivedEventType(BaseModel):
    class Meta:
        name = "AeGenericHWDataReceivedEventType"

    model_config = ConfigDict(defer_build=True)
    required_port_tref: list[AeReference] = field(
        default_factory=list,
        metadata={
            "name": "REQUIRED-PORT-TREF",
            "type": "Element",
            "min_occurs": 1,
        },
    )


class AeGenericHwtimingEventType(BaseModel):
    class Meta:
        name = "AeGenericHWTimingEventType"

    model_config = ConfigDict(defer_build=True)
    period: "AeGenericHwtimingEventType.Period" = field(
        metadata={
            "name": "PERIOD",
            "type": "Element",
            "required": True,
        }
    )

    class Period(BaseModel):
        model_config = ConfigDict(defer_build=True)
        value: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )
        unit: PeriodUnit = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )


class AeGenericHwtriggerEventType(BaseModel):
    class Meta:
        name = "AeGenericHWTriggerEventType"

    model_config = ConfigDict(defer_build=True)
    trigger: AeReference = field(
        metadata={
            "name": "TRIGGER",
            "type": "Element",
            "required": True,
        }
    )


class AeInterfaceData(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    accessed_variable: "AeInterfaceData.AccessedVariable" = field(
        metadata={
            "name": "ACCESSED-VARIABLE",
            "type": "Element",
            "required": True,
        }
    )

    class AccessedVariable(BaseModel):
        model_config = ConfigDict(defer_build=True)
        autosar_variable_iref: "AeInterfaceData.AccessedVariable.AutosarVariableIref" = field(
            metadata={
                "name": "AUTOSAR-VARIABLE-IREF",
                "type": "Element",
                "required": True,
            }
        )

        class AutosarVariableIref(BaseModel):
            """
            :ivar port_prototype_ref:
            :ivar target_data_prototype_ref: Example /Sender-Receiver-
                Interface/DataElment
            """

            model_config = ConfigDict(defer_build=True)
            port_prototype_ref: AeReference = field(
                metadata={
                    "name": "PORT-PROTOTYPE-REF",
                    "type": "Element",
                    "required": True,
                }
            )
            target_data_prototype_ref: AeReference = field(
                metadata={
                    "name": "TARGET-DATA-PROTOTYPE-REF",
                    "type": "Element",
                    "required": True,
                }
            )


class AeInterfaceType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    data_elements: "AeInterfaceType.DataElements" = field(
        metadata={
            "name": "DATA-ELEMENTS",
            "type": "Element",
            "required": True,
        }
    )

    class DataElements(BaseModel):
        model_config = ConfigDict(defer_build=True)
        variable_data_prototype: list[
            "AeInterfaceType.DataElements.VariableDataPrototype"
        ] = field(
            default_factory=list,
            metadata={
                "name": "VARIABLE-DATA-PROTOTYPE",
                "type": "Element",
                "min_occurs": 1,
            },
        )

        class VariableDataPrototype(BaseModel):
            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            type_tref: "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref" = field(
                metadata={
                    "name": "TYPE-TREF",
                    "type": "Element",
                    "required": True,
                }
            )

            class TypeTref(BaseModel):
                model_config = ConfigDict(defer_build=True)
                array_of_uint8: Optional[
                    "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfUint8"
                ] = field(
                    default=None,
                    metadata={
                        "name": "Array-of-uint8",
                        "type": "Element",
                    },
                )
                array_of_float: Optional[
                    "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfFloat"
                ] = field(
                    default=None,
                    metadata={
                        "name": "Array-of-float",
                        "type": "Element",
                    },
                )

                class ArrayOfUint8(BaseModel):
                    """
                    :ivar random_values_generated: support ranges of the
                        form [ min : max ] where min and max define a
                        range of randomly generated values determined
                    :ivar fixed_values_generated:
                    :ivar no_of_bytes:
                    """

                    model_config = ConfigDict(defer_build=True)
                    random_values_generated: Optional[
                        "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfUint8.RandomValuesGenerated"
                    ] = field(
                        default=None,
                        metadata={
                            "name": "Random-Values-generated",
                            "type": "Element",
                        },
                    )
                    fixed_values_generated: Optional[
                        "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfUint8.FixedValuesGenerated"
                    ] = field(
                        default=None,
                        metadata={
                            "name": "Fixed-values-generated",
                            "type": "Element",
                        },
                    )
                    no_of_bytes: int = field(
                        metadata={
                            "name": "No_Of_Bytes",
                            "type": "Attribute",
                            "required": True,
                            "min_inclusive": 1,
                            "max_inclusive": 130023424,
                        }
                    )

                    class RandomValuesGenerated(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        data_range: Optional[
                            "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfUint8.RandomValuesGenerated.DataRange"
                        ] = field(
                            default=None,
                            metadata={
                                "name": "Data-Range",
                                "type": "Element",
                            },
                        )

                        class DataRange(BaseModel):
                            model_config = ConfigDict(defer_build=True)
                            min: int = field(
                                metadata={
                                    "type": "Attribute",
                                    "required": True,
                                    "min_inclusive": 0,
                                    "max_inclusive": 255,
                                }
                            )
                            max: int = field(
                                metadata={
                                    "type": "Attribute",
                                    "required": True,
                                    "min_inclusive": 0,
                                    "max_inclusive": 255,
                                }
                            )

                    class FixedValuesGenerated(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        default_value: int = field(
                            metadata={
                                "name": "Default-Value",
                                "type": "Attribute",
                                "required": True,
                                "min_inclusive": 0,
                                "max_inclusive": 255,
                            }
                        )

                class ArrayOfFloat(BaseModel):
                    """
                    :ivar random_values_generated: support ranges of the
                        form [ min : max ] where min and max define a
                        range of randomly generated values determined
                    :ivar fixed_values_generated:
                    :ivar no_of_elements:
                    """

                    model_config = ConfigDict(defer_build=True)
                    random_values_generated: Optional[
                        "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfFloat.RandomValuesGenerated"
                    ] = field(
                        default=None,
                        metadata={
                            "name": "Random-Values-generated",
                            "type": "Element",
                        },
                    )
                    fixed_values_generated: Optional[
                        "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfFloat.FixedValuesGenerated"
                    ] = field(
                        default=None,
                        metadata={
                            "name": "Fixed-values-generated",
                            "type": "Element",
                        },
                    )
                    no_of_elements: int = field(
                        metadata={
                            "name": "No_Of_Elements",
                            "type": "Attribute",
                            "required": True,
                            "min_inclusive": 1,
                            "max_inclusive": 130023424,
                        }
                    )

                    class RandomValuesGenerated(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        data_range: Optional[
                            "AeInterfaceType.DataElements.VariableDataPrototype.TypeTref.ArrayOfFloat.RandomValuesGenerated.DataRange"
                        ] = field(
                            default=None,
                            metadata={
                                "name": "Data-Range",
                                "type": "Element",
                            },
                        )

                        class DataRange(BaseModel):
                            model_config = ConfigDict(defer_build=True)
                            min: float = field(
                                metadata={
                                    "type": "Attribute",
                                    "required": True,
                                    "min_inclusive": 0.0,
                                }
                            )
                            max: float = field(
                                metadata={
                                    "type": "Attribute",
                                    "required": True,
                                    "min_inclusive": 0.0,
                                }
                            )

                    class FixedValuesGenerated(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        default_value: float = field(
                            metadata={
                                "name": "Default-Value",
                                "type": "Attribute",
                                "required": True,
                                "min_inclusive": 0.0,
                            }
                        )


class AeNetworkTopologyType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    inter_ecu_communication: Optional[
        "AeNetworkTopologyType.InterEcuCommunication"
    ] = field(
        default=None,
        metadata={
            "name": "InterECU_communication",
            "type": "Element",
        },
    )

    class InterEcuCommunication(BaseModel):
        model_config = ConfigDict(defer_build=True)
        can_bus: list["AeNetworkTopologyType.InterEcuCommunication.CanBus"] = (
            field(
                default_factory=list,
                metadata={
                    "name": "CAN-BUS",
                    "type": "Element",
                    "max_occurs": 8,
                },
            )
        )
        eth_switch: list[
            "AeNetworkTopologyType.InterEcuCommunication.EthSwitch"
        ] = field(
            default_factory=list,
            metadata={
                "name": "Eth-Switch",
                "type": "Element",
                "max_occurs": 8,
            },
        )

        class CanBus(BaseModel):
            """
            :ivar short_name:
            :ivar baud_rate: CAN bus baudrate. Unit in bit per second
                (bps)
            :ivar interface_tref: Example /InterfaceName
            :ivar can_fd:
            """

            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            baud_rate: "AeNetworkTopologyType.InterEcuCommunication.CanBus.BaudRate" = field(
                metadata={
                    "name": "BaudRate",
                    "type": "Element",
                    "required": True,
                }
            )
            interface_tref: list[AeReference] = field(
                default_factory=list,
                metadata={
                    "name": "INTERFACE-TREF",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )
            can_fd: bool = field(
                default=True,
                metadata={
                    "name": "CAN-FD",
                    "type": "Attribute",
                },
            )

            class BaudRate(BaseModel):
                model_config = ConfigDict(defer_build=True)
                value: int = field(
                    default=500000,
                    metadata={
                        "type": "Attribute",
                    },
                )

        class EthSwitch(BaseModel):
            """
            :ivar short_name:
            :ivar interface_tref: Example /InterfaceName
            """

            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            interface_tref: list[AeReference] = field(
                default_factory=list,
                metadata={
                    "name": "INTERFACE-TREF",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )


class AeOperationType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    write: Optional["AeOperationType.Write"] = field(
        default=None,
        metadata={
            "name": "WRITE",
            "type": "Element",
        },
    )
    read: Optional["AeOperationType.Read"] = field(
        default=None,
        metadata={
            "name": "READ",
            "type": "Element",
        },
    )
    latency: Optional["AeOperationType.Latency"] = field(
        default=None,
        metadata={
            "name": "LATENCY",
            "type": "Element",
        },
    )
    load: Optional["AeOperationType.Load"] = field(
        default=None,
        metadata={
            "name": "LOAD",
            "type": "Element",
        },
    )
    custom_operation: Optional["AeOperationType.CustomOperation"] = field(
        default=None,
        metadata={
            "name": "CUSTOM-OPERATION",
            "type": "Element",
        },
    )
    batch_normalization: Optional["AeOperationType.BatchNormalization"] = (
        field(
            default=None,
            metadata={
                "name": "BATCH-NORMALIZATION",
                "type": "Element",
            },
        )
    )
    convolution: Optional["AeOperationType.Convolution"] = field(
        default=None,
        metadata={
            "name": "CONVOLUTION",
            "type": "Element",
        },
    )
    hardmax: Optional["AeOperationType.Hardmax"] = field(
        default=None,
        metadata={
            "name": "HARDMAX",
            "type": "Element",
        },
    )
    log_softmax: Optional["AeOperationType.LogSoftmax"] = field(
        default=None,
        metadata={
            "name": "LOG-SOFTMAX",
            "type": "Element",
        },
    )
    max_pool: Optional["AeOperationType.MaxPool"] = field(
        default=None,
        metadata={
            "name": "MAX-POOL",
            "type": "Element",
        },
    )
    avg_pool: Optional["AeOperationType.AvgPool"] = field(
        default=None,
        metadata={
            "name": "AVG-POOL",
            "type": "Element",
        },
    )
    transposed_convolution: Optional[
        "AeOperationType.TransposedConvolution"
    ] = field(
        default=None,
        metadata={
            "name": "TRANSPOSED-CONVOLUTION",
            "type": "Element",
        },
    )

    class Write(BaseModel):
        """For SWC-Custom-Behavior:
        DEST=/swc-name/internalBehavior-name/runnable-name/variableAccess-name
        For Generic_Hardware:
        DEST=/genericHardware-name/pPort-name"""

        model_config = ConfigDict(defer_build=True)
        iref: AeReference = field(
            metadata={
                "name": "IREF",
                "type": "Element",
                "required": True,
            }
        )

    class Read(BaseModel):
        """For SWC-Custom-Behavior:
        DEST=/swc-name/internalBehavior-name/runnable-name/variableAccess-name
        For Generic_Hardware:
        DEST=/genericHardware-name/rPort-name"""

        model_config = ConfigDict(defer_build=True)
        iref: AeReference = field(
            metadata={
                "name": "IREF",
                "type": "Element",
                "required": True,
            }
        )

    class Latency(BaseModel):
        """Time Value of type unsigned long and unit of {s, ms, us, ns}.

        Note: ONLY used with Generic Hardware
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )
        unit: LatencyUnit = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )

    class Load(BaseModel):
        """
        Number of cycles value of type unsigned int.
        """

        model_config = ConfigDict(defer_build=True)
        value: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )

    class CustomOperation(BaseModel):
        """
        Custom operation to be executed.

        :ivar function_prototype: Function prototype to be executed, it
            must be in the following format: "void func_name(void)"
        :ivar header_file: Header file -that has the function prototype-
            to be included.
        :ivar includes_dir: Path to the includes directory.
        :ivar sources_dir: Path to the sources directory.
        """

        model_config = ConfigDict(defer_build=True)
        function_prototype: str = field(
            metadata={
                "name": "functionPrototype",
                "type": "Attribute",
                "required": True,
            }
        )
        header_file: str = field(
            metadata={
                "name": "headerFile",
                "type": "Attribute",
                "required": True,
            }
        )
        includes_dir: str = field(
            metadata={
                "name": "includesDir",
                "type": "Attribute",
                "required": True,
            }
        )
        sources_dir: str = field(
            metadata={
                "name": "sourcesDir",
                "type": "Attribute",
                "required": True,
            }
        )

    class BatchNormalization(BaseModel):
        """
        Batch Normalization Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )

    class Convolution(BaseModel):
        """
        Convolution Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )
        kernel_size: int = field(
            default=3,
            metadata={
                "name": "kernelSize",
                "type": "Attribute",
            },
        )

    class Hardmax(BaseModel):
        """
        Hardmax Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )

    class LogSoftmax(BaseModel):
        """
        Logarithm of Softmax Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )

    class MaxPool(BaseModel):
        """
        Max Pooling Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )
        kernel_size: int = field(
            default=3,
            metadata={
                "name": "kernelSize",
                "type": "Attribute",
            },
        )
        stride: int = field(
            default=2,
            metadata={
                "type": "Attribute",
            },
        )

    class AvgPool(BaseModel):
        """
        Max Pooling Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )
        kernel_size: int = field(
            default=3,
            metadata={
                "name": "kernelSize",
                "type": "Attribute",
            },
        )
        stride: int = field(
            default=2,
            metadata={
                "type": "Attribute",
            },
        )

    class TransposedConvolution(BaseModel):
        """
        Transposed Convolution Operation.
        """

        model_config = ConfigDict(defer_build=True)
        height: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        width: int = field(
            default=1024,
            metadata={
                "type": "Attribute",
            },
        )
        channels: int = field(
            default=3,
            metadata={
                "type": "Attribute",
            },
        )
        kernel_size: int = field(
            default=3,
            metadata={
                "name": "kernelSize",
                "type": "Attribute",
            },
        )


class AePreBuiltApplicationType(BaseModel):
    """
    :ivar short_name:
    :ivar path: Path to a pre-built application to run over the hardware
    """

    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    path: AeReference = field(
        metadata={
            "name": "PATH",
            "type": "Element",
            "required": True,
        }
    )


class AeTimeEventType(BaseModel):
    """
    :ivar short_name:
    :ivar start_on_event_ref: Example /SWC1/SWC1Behav/Runnable
    :ivar custom_behavior_ref: Example /Behavior1
    :ivar period:
    """

    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    start_on_event_ref: AeReference = field(
        metadata={
            "name": "START-ON-EVENT-REF",
            "type": "Element",
            "required": True,
        }
    )
    custom_behavior_ref: Optional[AeReference] = field(
        default=None,
        metadata={
            "name": "CUSTOM-BEHAVIOR-REF",
            "type": "Element",
        },
    )
    period: "AeTimeEventType.Period" = field(
        metadata={
            "name": "PERIOD",
            "type": "Element",
            "required": True,
        }
    )

    class Period(BaseModel):
        model_config = ConfigDict(defer_build=True)
        value: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )
        unit: PeriodUnit = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )


class AeDualCoreCluster(AeCpuCluster):
    model_config = ConfigDict(defer_build=True)
    cores_per_cluster: AeDualCoreClusterCoresPerCluster = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeHexaCoreCluster(AeCpuCluster):
    model_config = ConfigDict(defer_build=True)
    cores_per_cluster: AeHexaCoreClusterCoresPerCluster = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeOctaCoreCluster(AeCpuCluster):
    model_config = ConfigDict(defer_build=True)
    cores_per_cluster: AeOctaCoreClusterCoresPerCluster = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeOperationsSequenceType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    operation: list[AeOperationType] = field(
        default_factory=list,
        metadata={
            "name": "OPERATION",
            "type": "Element",
            "min_occurs": 1,
        },
    )


class AeQuadCoreCluster(AeCpuCluster):
    model_config = ConfigDict(defer_build=True)
    cores_per_cluster: AeQuadCoreClusterCoresPerCluster = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeSingleCoreCluster(AeCpuCluster):
    model_config = ConfigDict(defer_build=True)
    cores_per_cluster: AeSingleCoreClusterCoresPerCluster = field(
        metadata={
            "name": "CoresPerCluster",
            "type": "Attribute",
            "required": True,
        }
    )


class AeSwcType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    ports: Optional["AeSwcType.Ports"] = field(
        default=None,
        metadata={
            "name": "PORTS",
            "type": "Element",
        },
    )
    internal_behaviors: "AeSwcType.InternalBehaviors" = field(
        metadata={
            "name": "INTERNAL-BEHAVIORS",
            "type": "Element",
            "required": True,
        }
    )

    class Ports(BaseModel):
        model_config = ConfigDict(defer_build=True)
        p_port_prototype: list["AeSwcType.Ports.PPortPrototype"] = field(
            default_factory=list,
            metadata={
                "name": "P-PORT-PROTOTYPE",
                "type": "Element",
                "sequence": 1,
            },
        )
        r_port_prototype: list["AeSwcType.Ports.RPortPrototype"] = field(
            default_factory=list,
            metadata={
                "name": "R-PORT-PROTOTYPE",
                "type": "Element",
                "sequence": 1,
            },
        )

        class PPortPrototype(BaseModel):
            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            provided_interface_tref: AeReference = field(
                metadata={
                    "name": "PROVIDED-INTERFACE-TREF",
                    "type": "Element",
                    "required": True,
                }
            )

        class RPortPrototype(BaseModel):
            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            required_interface_tref: AeReference = field(
                metadata={
                    "name": "REQUIRED-INTERFACE-TREF",
                    "type": "Element",
                    "required": True,
                }
            )

    class InternalBehaviors(BaseModel):
        model_config = ConfigDict(defer_build=True)
        swc_internal_behavior: "AeSwcType.InternalBehaviors.SwcInternalBehavior" = field(
            metadata={
                "name": "SWC-INTERNAL-BEHAVIOR",
                "type": "Element",
                "required": True,
            }
        )

        class SwcInternalBehavior(BaseModel):
            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            runnables: "AeSwcType.InternalBehaviors.SwcInternalBehavior.Runnables" = field(
                metadata={
                    "name": "RUNNABLES",
                    "type": "Element",
                    "required": True,
                }
            )
            events: "AeSwcType.InternalBehaviors.SwcInternalBehavior.Events" = field(
                metadata={
                    "name": "EVENTS",
                    "type": "Element",
                    "required": True,
                }
            )

            class Runnables(BaseModel):
                model_config = ConfigDict(defer_build=True)
                runnable_entity: list[
                    "AeSwcType.InternalBehaviors.SwcInternalBehavior.Runnables.RunnableEntity"
                ] = field(
                    default_factory=list,
                    metadata={
                        "name": "RUNNABLE-ENTITY",
                        "type": "Element",
                        "min_occurs": 1,
                    },
                )

                class RunnableEntity(BaseModel):
                    model_config = ConfigDict(defer_build=True)
                    short_name: AeIdentity = field(
                        metadata={
                            "name": "SHORT-NAME",
                            "type": "Element",
                            "required": True,
                        }
                    )
                    data_read_access: list[
                        "AeSwcType.InternalBehaviors.SwcInternalBehavior.Runnables.RunnableEntity.DataReadAccess"
                    ] = field(
                        default_factory=list,
                        metadata={
                            "name": "DATA-READ-ACCESS",
                            "type": "Element",
                        },
                    )
                    data_write_access: list[
                        "AeSwcType.InternalBehaviors.SwcInternalBehavior.Runnables.RunnableEntity.DataWriteAccess"
                    ] = field(
                        default_factory=list,
                        metadata={
                            "name": "DATA-WRITE-ACCESS",
                            "type": "Element",
                        },
                    )

                    class DataReadAccess(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        variable_access: AeInterfaceData = field(
                            metadata={
                                "name": "VARIABLE-ACCESS",
                                "type": "Element",
                                "required": True,
                            }
                        )

                    class DataWriteAccess(BaseModel):
                        model_config = ConfigDict(defer_build=True)
                        variable_access: AeInterfaceData = field(
                            metadata={
                                "name": "VARIABLE-ACCESS",
                                "type": "Element",
                                "required": True,
                            }
                        )

            class Events(BaseModel):
                model_config = ConfigDict(defer_build=True)
                timing_event: list[AeTimeEventType] = field(
                    default_factory=list,
                    metadata={
                        "name": "TIMING-EVENT",
                        "type": "Element",
                    },
                )
                data_received_event: list[AeDataReceivedEventType] = field(
                    default_factory=list,
                    metadata={
                        "name": "DATA-RECEIVED-EVENT",
                        "type": "Element",
                    },
                )


class AeUcieInterfaceType(BaseModel):
    """
    Interchiplet communication interface over UCIe.
    """

    model_config = ConfigDict(defer_build=True)
    endpoint_dma_configuration: Optional[AeEndpointDmaconfigType] = field(
        default=None,
        metadata={
            "name": "Endpoint_DMA_Configuration",
            "type": "Element",
        },
    )
    d2_d_configuration: list[AeD2DconfigType] = field(
        default_factory=list,
        metadata={
            "name": "D2D_Configuration",
            "type": "Element",
            "max_occurs": 3,
        },
    )
    mode: AeUcieInterfaceTypeMode = field(
        metadata={
            "name": "Mode",
            "type": "Attribute",
            "required": True,
        }
    )


class AeCpuArchType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    operating_system: AeOsType = field(
        metadata={
            "name": "Operating-System",
            "type": "Element",
            "required": True,
        }
    )
    armv9_family: Optional["AeCpuArchType.Armv9Family"] = field(
        default=None,
        metadata={
            "name": "ARMV9-Family",
            "type": "Element",
        },
    )
    armv8_family: Optional["AeCpuArchType.Armv8Family"] = field(
        default=None,
        metadata={
            "name": "ARMV8-Family",
            "type": "Element",
        },
    )
    armv7_family: Optional["AeCpuArchType.Armv7Family"] = field(
        default=None,
        metadata={
            "name": "ARMV7-Family",
            "type": "Element",
        },
    )

    class Armv9Family(BaseModel):
        model_config = ConfigDict(defer_build=True)
        afm_cortex_a510: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "AFM-CortexA510",
                "type": "Element",
            },
        )
        afm_cortex_a710: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "AFM-CortexA710",
                "type": "Element",
            },
        )
        afm_cortex_a720: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "AFM-CortexA720",
                "type": "Element",
            },
        )

    class Armv8Family(BaseModel):
        model_config = ConfigDict(defer_build=True)
        cortex_a53: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "CortexA53",
                "type": "Element",
            },
        )
        cortex_a57: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "CortexA57",
                "type": "Element",
            },
        )
        cortex_a72: Optional[AeOctaCoreCluster] = field(
            default=None,
            metadata={
                "name": "CortexA72",
                "type": "Element",
            },
        )
        cortex_r52: Optional[AeQuadCoreCluster] = field(
            default=None,
            metadata={
                "name": "CortexR52",
                "type": "Element",
            },
        )

    class Armv7Family(BaseModel):
        model_config = ConfigDict(defer_build=True)
        cortex_m7: AeSingleCoreCluster = field(
            metadata={
                "name": "CortexM7",
                "type": "Element",
                "required": True,
            }
        )


class AeGenericHardwareType(BaseModel):
    """Generic Hardware component.

    Its behavior can be configured to be:
    - Producer: Generates Data and send to required Destination
    - Consumer: Reads Data from specified source
    - Normal: Reads Data from specified source, process it then send to required Destination
    It has two operation modes:
    - Periodic
    - Event triggered
    After the end of its behavior it raises an interrupt, it must be handled by the core the HW is linked to.

    :ivar short_name:
    :ivar frequency:
    :ivar axi_master_port:
    :ivar memory_interface: Specifies whether Memory Interface is done
        through Internal Memory or Shared Memory.
    :ivar internal_behavior:
    :ivar power_parameters:
    """

    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    frequency: AeGenericHwfrequency = field(
        metadata={
            "name": "Frequency",
            "type": "Element",
            "required": True,
        }
    )
    axi_master_port: "AeGenericHardwareType.AxiMasterPort" = field(
        metadata={
            "name": "AXI-Master-Port",
            "type": "Element",
            "required": True,
        }
    )
    memory_interface: "AeGenericHardwareType.MemoryInterface" = field(
        metadata={
            "name": "MEMORY-INTERFACE",
            "type": "Element",
            "required": True,
        }
    )
    internal_behavior: "AeGenericHardwareType.InternalBehavior" = field(
        metadata={
            "name": "INTERNAL-BEHAVIOR",
            "type": "Element",
            "required": True,
        }
    )
    power_parameters: Optional[AePowerParameterType] = field(
        default=None,
        metadata={
            "name": "POWER-PARAMETERS",
            "type": "Element",
        },
    )

    class AxiMasterPort(BaseModel):
        """
        :ivar priority: Priority range from 0 to 255. 0 is the highest
            priority and given to the CPU connected to the AXI bus. 255
            is the lowest priority.
        """

        model_config = ConfigDict(defer_build=True)
        priority: int = field(
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )

    class MemoryInterface(BaseModel):
        """
        :ivar internal_memory: To use Internal Memory set
            InternalMemory="true" Otherwise Shared Memory is used.
        """

        model_config = ConfigDict(defer_build=True)
        internal_memory: bool = field(
            default=False,
            metadata={
                "name": "InternalMemory",
                "type": "Attribute",
            },
        )

    class InternalBehavior(BaseModel):
        model_config = ConfigDict(defer_build=True)
        ports: "AeGenericHardwareType.InternalBehavior.Ports" = field(
            metadata={
                "name": "PORTS",
                "type": "Element",
                "required": True,
            }
        )
        operations_sequence: AeOperationsSequenceType = field(
            metadata={
                "name": "OPERATIONS-SEQUENCE",
                "type": "Element",
                "required": True,
            }
        )
        event: "AeGenericHardwareType.InternalBehavior.Event" = field(
            metadata={
                "name": "EVENT",
                "type": "Element",
                "required": True,
            }
        )

        class Ports(BaseModel):
            model_config = ConfigDict(defer_build=True)
            p_port_prototype: list[
                "AeGenericHardwareType.InternalBehavior.Ports.PPortPrototype"
            ] = field(
                default_factory=list,
                metadata={
                    "name": "P-PORT-PROTOTYPE",
                    "type": "Element",
                    "sequence": 1,
                },
            )
            r_port_prototype: list[
                "AeGenericHardwareType.InternalBehavior.Ports.RPortPrototype"
            ] = field(
                default_factory=list,
                metadata={
                    "name": "R-PORT-PROTOTYPE",
                    "type": "Element",
                    "sequence": 1,
                },
            )

            class PPortPrototype(BaseModel):
                model_config = ConfigDict(defer_build=True)
                short_name: AeIdentity = field(
                    metadata={
                        "name": "SHORT-NAME",
                        "type": "Element",
                        "required": True,
                    }
                )
                provided_interface_tref: AeReference = field(
                    metadata={
                        "name": "PROVIDED-INTERFACE-TREF",
                        "type": "Element",
                        "required": True,
                    }
                )

            class RPortPrototype(BaseModel):
                model_config = ConfigDict(defer_build=True)
                short_name: AeIdentity = field(
                    metadata={
                        "name": "SHORT-NAME",
                        "type": "Element",
                        "required": True,
                    }
                )
                required_interface_tref: AeReference = field(
                    metadata={
                        "name": "REQUIRED-INTERFACE-TREF",
                        "type": "Element",
                        "required": True,
                    }
                )

        class Event(BaseModel):
            model_config = ConfigDict(defer_build=True)
            timing_event: Optional[AeGenericHwtimingEventType] = field(
                default=None,
                metadata={
                    "name": "TIMING-EVENT",
                    "type": "Element",
                },
            )
            data_received_event: Optional[AeGenericHwdataReceivedEventType] = (
                field(
                    default=None,
                    metadata={
                        "name": "DATA-RECEIVED-EVENT",
                        "type": "Element",
                    },
                )
            )
            trigger_event: Optional[AeGenericHwtriggerEventType] = field(
                default=None,
                metadata={
                    "name": "TRIGGER-EVENT",
                    "type": "Element",
                },
            )


class AeSwcCustomBehaviorType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    operations_sequence: AeOperationsSequenceType = field(
        metadata={
            "name": "OPERATIONS-SEQUENCE",
            "type": "Element",
            "required": True,
        }
    )


class AeChipletType(BaseModel):
    model_config = ConfigDict(defer_build=True)
    short_name: AeIdentity = field(
        metadata={
            "name": "SHORT-NAME",
            "type": "Element",
            "required": True,
        }
    )
    axi_bus: AeAxiBusType = field(
        metadata={
            "name": "AXI-BUS",
            "type": "Element",
            "required": True,
        }
    )
    ethernet_interface: AeEthernetInterfaceType = field(
        metadata={
            "name": "ETHERNET-INTERFACE",
            "type": "Element",
            "required": True,
        }
    )
    ucie_interface: AeUcieInterfaceType = field(
        metadata={
            "name": "UCIe-INTERFACE",
            "type": "Element",
            "required": True,
        }
    )
    cpu_cluster: Optional[AeCpuArchType] = field(
        default=None,
        metadata={
            "name": "CPU_Cluster",
            "type": "Element",
        },
    )
    generic_hardware: list[AeGenericHardwareType] = field(
        default_factory=list,
        metadata={
            "name": "Generic_Hardware",
            "type": "Element",
        },
    )


class ArPackage(BaseModel):
    class Meta:
        name = "AR-PACKAGE"

    model_config = ConfigDict(defer_build=True)
    elements: list["ArPackage.Elements"] = field(
        default_factory=list,
        metadata={
            "name": "ELEMENTS",
            "type": "Element",
            "min_occurs": 1,
        },
    )

    class Elements(BaseModel):
        model_config = ConfigDict(defer_build=True)
        simulation_time: Optional["ArPackage.Elements.SimulationTime"] = field(
            default=None,
            metadata={
                "name": "Simulation-Time",
                "type": "Element",
            },
        )
        sender_receiver_interface: list[AeInterfaceType] = field(
            default_factory=list,
            metadata={
                "name": "SENDER-RECEIVER-INTERFACE",
                "type": "Element",
            },
        )
        application_sw_component_type: list[AeSwcType] = field(
            default_factory=list,
            metadata={
                "name": "APPLICATION-SW-COMPONENT-TYPE",
                "type": "Element",
            },
        )
        swc_custom_behavior: list[AeSwcCustomBehaviorType] = field(
            default_factory=list,
            metadata={
                "name": "SWC-CUSTOM-BEHAVIOR",
                "type": "Element",
            },
        )
        pre_built_application: list[AePreBuiltApplicationType] = field(
            default_factory=list,
            metadata={
                "name": "PRE-BUILT-APPLICATION",
                "type": "Element",
            },
        )
        network_topology: Optional[AeNetworkTopologyType] = field(
            default=None,
            metadata={
                "name": "Network-Topology",
                "type": "Element",
            },
        )
        ecus: list["ArPackage.Elements.Ecus"] = field(
            default_factory=list,
            metadata={
                "name": "ECUs",
                "type": "Element",
                "min_occurs": 1,
                "max_occurs": 100,
            },
        )
        hw_sw_mapping: list[AeHwSwMappingType] = field(
            default_factory=list,
            metadata={
                "name": "HW-SW-MAPPING",
                "type": "Element",
            },
        )
        analysis: Optional[AeAnalysisType] = field(
            default=None,
            metadata={
                "name": "Analysis",
                "type": "Element",
            },
        )

        class SimulationTime(BaseModel):
            model_config = ConfigDict(defer_build=True)
            value: int = field(
                metadata={
                    "type": "Attribute",
                    "required": True,
                }
            )
            unit: SimulationTimeUnit = field(
                metadata={
                    "type": "Attribute",
                    "required": True,
                }
            )

        class Ecus(BaseModel):
            """
            :ivar short_name:
            :ivar so_cs: Notes: - Total number of all SoCs in the System
                is max equal 100 - Total number of all
                CPU_Cluster/Chiplets in each SoC is max equal 4 - Total
                number of all Generic_Hardware in each Chiplets/SoC is
                max equal 32
            """

            model_config = ConfigDict(defer_build=True)
            short_name: AeIdentity = field(
                metadata={
                    "name": "SHORT-NAME",
                    "type": "Element",
                    "required": True,
                }
            )
            so_cs: list["ArPackage.Elements.Ecus.SoCs"] = field(
                default_factory=list,
                metadata={
                    "name": "SoCs",
                    "type": "Element",
                    "min_occurs": 1,
                    "max_occurs": 100,
                },
            )

            class SoCs(BaseModel):
                model_config = ConfigDict(defer_build=True)
                short_name: AeIdentity = field(
                    metadata={
                        "name": "SHORT-NAME",
                        "type": "Element",
                        "required": True,
                    }
                )
                axi_bus: AeAxiBusType = field(
                    metadata={
                        "name": "AXI-BUS",
                        "type": "Element",
                        "required": True,
                    }
                )
                ethernet_interface: AeEthernetInterfaceType = field(
                    metadata={
                        "name": "ETHERNET-INTERFACE",
                        "type": "Element",
                        "required": True,
                    }
                )
                ucie_interface: AeUcieInterfaceType = field(
                    metadata={
                        "name": "UCIe-INTERFACE",
                        "type": "Element",
                        "required": True,
                    }
                )
                chiplet: list[AeChipletType] = field(
                    default_factory=list,
                    metadata={
                        "name": "Chiplet",
                        "type": "Element",
                    },
                )
                cpu_cluster: Optional[AeCpuArchType] = field(
                    default=None,
                    metadata={
                        "name": "CPU_Cluster",
                        "type": "Element",
                    },
                )
                generic_hardware: list[AeGenericHardwareType] = field(
                    default_factory=list,
                    metadata={
                        "name": "Generic_Hardware",
                        "type": "Element",
                    },
                )
